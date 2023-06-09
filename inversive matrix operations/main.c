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

// Function to read a matrix from a file
Matrix* readMatrixFromFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return NULL;
    }

    int size;
    fscanf(file, "%d", &size);  // Read the size of the matrix

    Matrix* matrix = allocateMatrix(size);

    // Read values for the matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fscanf(file, "%lf", &matrix->data[i][j]);
        }
    }

    fclose(file);
    return matrix;
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

// Function to compute the inverse of a matrix (placeholder implementation)
Matrix* calculateInverse(Matrix* matrix) {
    // Placeholder implementation: simply reverse the order of elements
    int size = matrix->size;
    Matrix* inverse = allocateMatrix(size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            inverse->data[i][j] = matrix->data[size - 1 - i][size - 1 - j];
        }
    }

    return inverse;
}

// Function to compute the sum of two matrices
Matrix* matrixSum(Matrix* matrix1, Matrix* matrix2) {
    int size = matrix1->size;
    Matrix* result = allocateMatrix(size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result->data[i][j] = matrix1->data[i][j] + matrix2->data[i][j];
        }
    }

    return result;
}

// Function to compute the product of a scalar and a matrix
Matrix* scalarProduct(double scalar, Matrix* matrix) {
    int size = matrix->size;
    Matrix* result = allocateMatrix(size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result->data[i][j] = scalar * matrix->data[i][j];
        }
    }

    return result;
}

int main() {
    Matrix* originalMatrix = readMatrixFromFile("input.txt");

    if (originalMatrix == NULL) {
        printf("Error reading the matrix from the file.\n");
        return 1;
    }

    Matrix* inverseMatrix = calculateInverse(originalMatrix);

    printf("Original Matrix:\n");
    printMatrix(originalMatrix);

    printf("\nInverse Matrix:\n");
    printMatrix(inverseMatrix);

    // Set a new value in the original matrix
    int rowIndex, colIndex;
    double newValue;
    printf("\nEnter the row index, column index, and new value to set in the original matrix: ");
    scanf("%d %d %lf", &rowIndex, &colIndex, &newValue);

    if (rowIndex >= 0 && rowIndex < originalMatrix->size && colIndex >= 0 && colIndex < originalMatrix->size) {
        // Update the original matrix
        originalMatrix->data[rowIndex][colIndex] = newValue;

        // Compute the difference matrix
        Matrix* differenceMatrix = allocateMatrix(originalMatrix->size);
        differenceMatrix->data[rowIndex][colIndex] = newValue - inverseMatrix->data[rowIndex][colIndex];

        // Compute the updated inverse matrix using the Sherman-Morrison formula
        Matrix* updatedInverseMatrix = matrixSum(inverseMatrix, scalarProduct(1 / (newValue - inverseMatrix->data[rowIndex][colIndex]), matrixSum(matrixSum(matrixSum(matrixSum(inverseMatrix, differenceMatrix), scalarProduct(-1, inverseMatrix)), scalarProduct(-1, differenceMatrix)), differenceMatrix)));

        printf("\nUpdated Original Matrix:\n");
        printMatrix(originalMatrix);

        printf("\nUpdated Inverse Matrix:\n");
        printMatrix(updatedInverseMatrix);

        freeMatrix(differenceMatrix);
        freeMatrix(updatedInverseMatrix);
    } else {
        printf("Invalid row or column index.\n");
    }

    freeMatrix(originalMatrix);
    freeMatrix(inverseMatrix);

    return 0;
}
