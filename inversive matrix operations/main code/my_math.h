#include <stdio.h>
#include <stdlib.h>
char my_type[] = "%Lf";
#define MAX_SIZE 10

void my_printMatrix(long double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%Lf ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

long double **my_allocateMatrix(int rows, int cols) {
    long double **matrix = malloc(rows * sizeof(long double *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(long double));
    }
    return matrix;
}

void my_freeMatrix(long double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void my_free_double_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

long double **my_invertMatrix(long double **matrix, int size) {
    // Create an identity matrix
    long double **identity = my_allocateMatrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            identity[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Perform Gauss-Jordan elimination with row swaps
    for (int i = 0; i < size; i++) {
        // Find the pivot element
        long double pivot = matrix[i][i];

        // If the pivot is zero, try to swap with a non-zero element below
        if (pivot == 0.0) {
            int rowToSwap = i + 1;
            while (rowToSwap < size && matrix[rowToSwap][i] == 0.0) {
                rowToSwap++;
            }

            if (rowToSwap < size) {
                // Swap rows
                long double *tempRow = matrix[i];
                matrix[i] = matrix[rowToSwap];
                matrix[rowToSwap] = tempRow;

                tempRow = identity[i];
                identity[i] = identity[rowToSwap];
                identity[rowToSwap] = tempRow;

                pivot = matrix[i][i];
            } else {
                // Matrix is singular, cannot be inverted
                printf("Matrix is singular. Cannot calculate inverse.\n");
                my_freeMatrix(identity, size);
                return NULL;
            }
        }

        // Scale the pivot row
        for (int j = 0; j < size; j++) {
            matrix[i][j] /= pivot;
            identity[i][j] /= pivot;
        }

        // Eliminate other rows
        for (int k = 0; k < size; k++) {
            if (k != i) {
                long double factor = matrix[k][i];
                for (int j = 0; j < size; j++) {
                    matrix[k][j] -= factor * matrix[i][j];
                    identity[k][j] -= factor * identity[i][j];
                }
            }
        }
    }
    my_printMatrix(identity, size);
    return identity;
}

void copyMatrix(long double **src, long double **dest, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

void copy_matrix(long double **src, double **dest, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i][j] = (double) src[i][j];
        }
    }
}
