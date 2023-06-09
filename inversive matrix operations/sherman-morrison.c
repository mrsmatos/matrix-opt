#include <stdio.h>
#include <stdlib.h>

/*
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

// Function to perform the Sherman-Morrison operation on two matrices
Matrix* shermanMorrison(Matrix* matrix1, Matrix* matrix2) {
    int size = matrix1->size;
    Matrix* result = allocateMatrix(size);

    // Perform the Sherman-Morrison operation
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result->data[i][j] = matrix1->data[i][j] - (matrix1->data[i][j] * matrix2->data[j][i] * matrix1->data[i][j]) /
                                                     (1 + matrix1->data[i][j] * matrix2->data[i][j]);
        }
    }

    return result;
}
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to allocate memory for a matrix
double** allocateMatrix(int rows, int columns) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        matrix[i] = (double*)malloc(columns * sizeof(double));
    return matrix;
}

// Function to free memory allocated for a matrix
void freeMatrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++)
        free(matrix[i]);
    free(matrix);
}

// Function to print a matrix
void printMatrix(double** matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++)
            printf("%.2f ", matrix[i][j]);
        printf("\n");
    }
}

// Function to update the inverse of a matrix using Sherman-Morrison formula
void shermanMorrisonInverse(double** matrix, double** inverse, int size, int row, int column, double value) {
    double** u = allocateMatrix(size, size);
    double** v = allocateMatrix(size, size);
    double factor, denominator;

    // Calculate the factor and denominator for the formula
    factor = 1.0 / (1.0 + value * inverse[column][row]);
    denominator = 1.0 / value;

    // Update the inverse matrix
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            u[i][j] = inverse[i][j] - factor * inverse[i][row] * inverse[column][j];

    // Update the inverse matrix further
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            v[i][j] = u[i][j] + (factor * u[row][i] * u[j][column]) / denominator;

    // Copy the updated inverse matrix back to the original matrix
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            inverse[i][j] = v[i][j];

    // Free the dynamically allocated memory
    freeMatrix(u, size);
    freeMatrix(v, size);
}

int main( int argc, char *argv[] ){
    int size;
    int row = 1;
    int column = 1;
    double value = 1.0;

    // Initialize size of original matrix
    while (scanf("%d", &size)!=EOF){
        // Allocate memory for the original matrix and inverse matrix
        double** matrix = allocateMatrix(size, size);
        double** inverse = allocateMatrix(size, size);

        // Initialize the original matrix
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                scanf("%lf", &matrix[i][j]);

        // Initialize size of inverse matrix
        scanf("%d", &size);

        // Initialize the inverse matrix
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                scanf("%lf", &inverse[i][j]);

        // get index and value of inverse matrix that will be changed
        scanf("%d %d %lf", &row, &column, &value);

        printf("Original Matrix:\n");
        printMatrix(matrix, size, size);

        printf("\nOriginal Inverse Matrix:\n");
        printMatrix(inverse, size, size);

        shermanMorrisonInverse(matrix, inverse, size, row, column, value);

        printf("\nUpdated Inverse Matrix:\n");
        printMatrix(inverse, size, size);

        // Free the dynamically allocated memory
        freeMatrix(matrix, size);
        freeMatrix(inverse, size);
    }
    return 0;
}
