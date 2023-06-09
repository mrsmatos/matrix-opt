#include <stdio.h>
#include <stdlib.h>

// Structure to hold a matrix
typedef struct {
    double** data;
    int size;
} Matrix;

// Function to allocate memory for a matrix
Matrix* allocateMatrix(int size) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->data = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix->data[i] = (double*)malloc(size * sizeof(double));
    }
    matrix->size = size;
    return matrix;
}

// Function to free memory allocated for a matrix
void freeMatrix(Matrix* matrix) {
    for (int i = 0; i < matrix->size; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

// Function to read matrix values from a file and put them into two dynamic square matrices
void readMatricesFromFile(const char* filename, Matrix** matrix1, Matrix** matrix2) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return;
    }

    int size;
    fscanf(file, "%d", &size);  // Read the size of the matrices

    *matrix1 = allocateMatrix(size);
    *matrix2 = allocateMatrix(size);

    // Read values for the first matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fscanf(file, "%lf", &((*matrix1)->data[i][j]));
        }
    }

    // Read values for the second matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fscanf(file, "%lf", &((*matrix2)->data[i][j]));
        }
    }

    fclose(file);
}

// Function to print a matrix
void printMatrix(Matrix* matrix) {
    for (int i = 0; i < matrix->size; i++) {
        for (int j = 0; j < matrix->size; j++) {
            printf("%.2f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

int main( int argc, char *argv[] )  {
    Matrix* matrix1;
    Matrix* matrix2;

    readMatricesFromFile("input.txt", &matrix1, &matrix2);

    printf("Matrix 1:\n");
    printMatrix(matrix1);

    printf("\nMatrix 2:\n");
    printMatrix(matrix2);

    freeMatrix(matrix1);
    freeMatrix(matrix2);

    return 0;
}
