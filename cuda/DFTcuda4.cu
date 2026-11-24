#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


#define PI 3.14159265358979323846f
#define NTHREAD 256

typedef struct {
    float real;
    float imag;
} MyComplex;

typedef struct {
    int width;
    int height;
    int max_value;
    unsigned char *data;
} PGMImage;

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

PGMImage readPGM(const char *filename) {
    PGMImage img;
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Errore nell'aprire il file");
        exit(1);
    }

    char format[3];
    fscanf(file, "%s", format);
    if (format[0] != 'P' || format[1] != '5') {
        fprintf(stderr, "Formato non supportato.\n");
        exit(1);
    }

    fscanf(file, "%d %d", &img.width, &img.height);
    fscanf(file, "%d", &img.max_value);
    fgetc(file);

    img.data = (unsigned char *)malloc(img.width * img.height);
    fread(img.data, 1, img.width * img.height, file);
    fclose(file);

    return img;
}

void writePGM(const char *filename, PGMImage img) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Errore nell'aprire il file per la scrittura");
        exit(1);
    }

    fprintf(file, "P5\n%d %d\n%d\n", img.width, img.height, img.max_value);
    fwrite(img.data, 1, img.width * img.height, file);
    fclose(file);
}

__global__ void dft2D(unsigned char *in, MyComplex *out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    MyComplex sum;
    sum.real = 0.0f;
    sum.imag = 0.0f;

    float angle, sinVal, cosVal;
    float pixel;

    __shared__ unsigned char in_share[NTHREAD];

    if (x < width && y < height) {
        for (int v = 0; v < height; v++) {
            for (int k = 0; k < width/NTHREAD ; k++) {
                
                in_share[blockDim.x * threadIdx.y + threadIdx.x ] = in[ v * width + k*NTHREAD + blockDim.x * threadIdx.y + threadIdx.x ];
                __syncthreads();

                for (int u = k*NTHREAD; u < (k+1)*NTHREAD ; u++){    
                    angle = -2.0f * PI * (fma((float)x, (float)u / width, (float)y * v / height));

                    __sincosf(angle, &sinVal, &cosVal);

                    pixel = in_share [u%NTHREAD];
                    sum.real = fmaf(pixel, cosVal, sum.real);
                    sum.imag = fmaf(pixel, sinVal, sum.imag);
                }
            }
        }
        out[y * width + x] = sum;
    }

}

__global__ void idft2D(MyComplex *in, unsigned char *out, int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    float angle, sinVal, cosVal;
    MyComplex coeff;

    __shared__ MyComplex in_share[NTHREAD];

    if (x < width && y < height) {
        for (int v = 0; v < height; v++) {
            for (int k = 0; k < width/NTHREAD ; k++) {
                
                in_share[blockDim.x * threadIdx.y + threadIdx.x ] = in[ v * width + k*NTHREAD + blockDim.x * threadIdx.y + threadIdx.x ];
                __syncthreads();

                for (int u = k*NTHREAD; u < (k+1)*NTHREAD ; u++){
                    angle = 2.0f * PI * (fma((float)x, (float)u / width, (float)y * v / height));

                    __sincosf(angle, &sinVal, &cosVal);

                    coeff = in_share[u%NTHREAD];
                    sum = fmaf(coeff.real, cosVal, sum);
                    sum = fmaf(-coeff.imag, sinVal, sum);
                }
            }
        }

        sum /= (width * height);
        out[y * width + x] = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
    }
}


__global__ void filtro(MyComplex *dft, int width, int height, float cutoff) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Xmezzi = width / 2;
    int Ymezzi = height / 2;
    float d;

    if (x < width && y < height) {
        
        d = fmaf(x-Xmezzi, x-Xmezzi, (y-Ymezzi)*(y-Ymezzi));

        if (d > cutoff) {
                dft[y * width + x].real = 0;
                dft[y * width + x].imag = 0;
        }
    }

}

int main() {
    const char *inputFile = "./Bologna-512.pgm";
    const char *outputFile = "output_cuda_4-512.pgm";
    float raggioFiltro = 260.0f; // 130 per 256 px    260 512 px


    PGMImage img = readPGM(inputFile);

    printf("Immagine caricata di dimensioni H:%d W:%d\n", img.height, img.width);

    unsigned char *d_in, *d_out;
    MyComplex *d_dft;
    CHECK(cudaMalloc(&d_in, img.width * img.height));
    CHECK(cudaMalloc(&d_out, img.width * img.height));
    CHECK(cudaMalloc(&d_dft, img.width * img.height * sizeof(MyComplex)));

    cudaMemcpy(d_in, img.data, img.width * img.height, cudaMemcpyHostToDevice);

    dim3 threadsDFT(16, 8);
    dim3 blocchiDFT((img.width + threadsDFT.x - 1) / threadsDFT.x, (img.height + threadsDFT.y - 1) / threadsDFT.y);

    dim3 threadsFiltro(16, 16); 
    dim3 blocchiFiltro((img.width + threadsFiltro.x - 1) / threadsFiltro.x, (img.height + threadsFiltro.y - 1) / threadsFiltro.y);               

    //-------------------------------------- DFT

    dft2D<<<blocchiDFT, threadsDFT>>>(d_in, d_dft, img.width, img.height);
    CHECK(cudaDeviceSynchronize());

    //-------------------------------------- 

    //-------------------------------------- FILTRO

    filtro<<<blocchiFiltro, threadsFiltro>>>(d_dft, img.width, img.height, raggioFiltro*raggioFiltro);
    CHECK(cudaDeviceSynchronize());

    //--------------------------------------

    //-------------------------------------- iDFT

    idft2D<<<blocchiDFT, threadsDFT>>>(d_dft, d_out, img.width, img.height);
    CHECK(cudaDeviceSynchronize());

    //-------------------------------------- 

    cudaMemcpy(img.data, d_out, img.width * img.height, cudaMemcpyDeviceToHost);

    writePGM(outputFile, img);


    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_dft);
    free(img.data);

    printf("Trasformata e antitrasformata completate. Risultato salvato in %s\n", outputFile);

    return 0;
}