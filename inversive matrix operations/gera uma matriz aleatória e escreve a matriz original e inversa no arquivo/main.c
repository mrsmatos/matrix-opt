#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

// Function to generate a random dynamic matrix
Matrix* generateRandomMatrix(int size) {
    Matrix* matrix = allocateMatrix(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            //matrix->data[i][j] = (double)rand() / RAND_MAX; // Assign a random value between 0 and 1
            matrix->data[i][j] = (double)(rand() % 100 + 1); // Assign a random value between 1 and 100
        }
    }
    return matrix;
}

/*
// Function to calculate the inverse of a matrix using the Sherman-Morrison formula
Matrix* calculateInverse(Matrix* matrix, int trash) {
    // Implement your inverse calculation logic here
    // This is just a placeholder implementation
    // Replace it with your own implementation

    int size = matrix->size;
    Matrix* inverse = allocateMatrix(size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            inverse->data[i][j] = matrix->data[size - i - 1][size - j - 1];
        }
    }

    return inverse;
}
*/

// Function to compute the inverse of a matrix using the Sherman-Morrison formula
Matrix* calculateInverse(Matrix* matrix) {
    int size = matrix->size;
    Matrix* inverse = allocateMatrix(size);

    // Compute the inverse matrix using the Sherman-Morrison formula
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            inverse->data[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int k = 0; k < size; k++) {
        double pivot = matrix->data[k][k];

        // Update the inverse matrix
        for (int i = 0; i < size; i++) {
            inverse->data[i][k] /= pivot;
        }

        // Update the matrix and inverse matrix using the Sherman-Morrison formula
        for (int i = 0; i < size; i++) {
            if (i != k) {
                double factor = matrix->data[i][k] / pivot;
                for (int j = 0; j < size; j++) {
                    matrix->data[i][j] -= factor * matrix->data[k][j];
                    inverse->data[i][j] -= factor * inverse->data[k][j];
                }
            }
        }
    }

    return inverse;
}

// Function to write a matrix to a file
void writeMatrixToFile(Matrix* matrix, const char* filename) {
    FILE* file = fopen(filename, "a");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return;
    }

    int size = matrix->size;
    fprintf(file, "%d\n", size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(file, "%.2f ", matrix->data[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main( int argc, char *argv[] )  {
    srand(time(NULL)); // Seed the random number generator with current time

    int size = 10; // Size of the matrix
    Matrix* originalMatrix = generateRandomMatrix(size);
    Matrix* inverseMatrix = calculateInverse(originalMatrix);

    writeMatrixToFile(originalMatrix, "original_inverse_matrix.txt");
    writeMatrixToFile(inverseMatrix, "original_inverse_matrix.txt");

    freeMatrix(originalMatrix);
    freeMatrix(inverseMatrix);

    return 0;
}
