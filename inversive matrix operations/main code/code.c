#include <stdio.h>
#include <stdlib.h>
#include "my_math.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>


// Structure to hold a matrix
typedef struct {
    long double** data;
    long double *u, *v, value_to_change;
    int i, j;
    int size;
} Matrix;

// Function to allocate memory for a matrix
Matrix* allocateMatrix(int size) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->data = (long double**)malloc(size * sizeof(long double*));
    matrix->u = (long double*)malloc(size * sizeof(long double));
    matrix->v = (long double*)malloc(size * sizeof(long double));
    for (int i = 0; i < size; i++) {
        matrix->data[i] = (long double*)malloc(size * sizeof(long double));
        matrix->u[i] = matrix->v[i] = 0.0; 
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
    free(matrix->u);
    free(matrix->v);
    free(matrix);
}

// Function to print a matrix
void printMatrix(Matrix* matrix) {
    for (int i = 0; i < matrix->size; i++) {
        for (int j = 0; j < matrix->size; j++) {
            printf("%Lf ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

// Function to read a matrix from keyboard (i.e. by scanf or input like file.in)
int read_matrix_from_keyboard(Matrix** matrix, Matrix** inverse_matrix) {
    int size;
    scanf("%d", &size);  // Read the size of the matrix
    
    (*matrix) = allocateMatrix(size);
    (*inverse_matrix) = allocateMatrix(size);
    if (matrix==NULL || inverse_matrix==NULL)
        return -1;
    
    (*matrix)->size = size; 
    (*inverse_matrix)->size = size;

    // Read values for the matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            scanf("%Lf", &(*matrix)->data[i][j]);
        }
    }

    scanf("%d", &size);  // Read the size of the inverse matrix
    // Read values for the inverse matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            scanf("%Lf", &(*inverse_matrix)->data[i][j]);
        }
    }

    //for (int i = 0; i < size; i++) scanf("%d", &matrix->u[i]);
    //for (int i = 0; i < size; i++) scanf("%d", &matrix->v[i]);

    // Read index and value to change in original matrix
    scanf("%d", &(*matrix)->i);
    scanf("%d", &(*matrix)->j);
    scanf("%Lf", &(*matrix)->value_to_change);
    return 0;
}

// Function to update the inverse matrix using Sherman-Morrison formula
// void updateInverseMatrix(int n, long double A[][n], long double invA[][n], long double v[], long double u[])
// ERROR: needs alpha, beta, gamma ;
void updateInverseMatrix(Matrix* matrix, Matrix* inverse_matrix) {
    int n = matrix->size;
    long double alpha = 1.0, beta = 1.0, gamma = 1.0;
    long double w[n][n];

    // Calculate the intermediate matrix w
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                w[i][j] = 1.0 / (matrix->v[i] - beta);
            else
                w[i][j] = 0.0;
        }
    }

    // Update the inverse matrix using Sherman-Morrison formula
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse_matrix->data[i][j] = inverse_matrix->data[i][j] - (matrix->u[i] * w[j][i]);
        }
    }

    // Calculate the intermediate values alpha and beta
    alpha = 1.0 / (1.0 - (beta * matrix->u[0]));
    beta = matrix->u[0];

    // Update the inverse matrix using Sherman-Morrison formula
    for (int i = 0; i < n; i++) {
        inverse_matrix->data[i][0] = alpha * inverse_matrix->data[i][0];
    }

    // Update the inverse matrix using Sherman-Morrison formula
    for (int i = 1; i < n; i++) {
        gamma = inverse_matrix->data[i][0];
        for (int j = 0; j < n; j++) {
            inverse_matrix->data[j][i] = inverse_matrix->data[j][i] + (gamma * matrix->u[j]);
        }
    }
}

// Function to update VALUE the inverse matrix using Sherman-Morrison formula
void updateInverse_ONE_VALUE(Matrix** original, Matrix** inverse, int row, int column, long double value) {
    long double alpha = ((long double) 1.0 ) / (value - (*original)->data[row][column]);
    
    // Update inverse matrix using Sherman-Morrison formula
    for (int i = 0; i < (*inverse)->size; i++) {
        for (int j = 0; j < (*inverse)->size; j++) {
            if (i == row && j == column) {
                (*inverse)->data[i][j] += alpha;
            } else {                
                (*inverse)->data[i][j] -= (alpha * (*original)->data[i][column] * (*inverse)->data[row][j]) / (1 + alpha * (*original)->data[row][column]);
            }
        }
    }
}

// Function to update ONE VECTOR the inverse matrix using Sherman-Morrison formula
void updateInverseMatrix_ONE_VECTOR(Matrix *inverse, long double *columnVector, int index) {
    int size = inverse->size;
    
    // Compute the intermediate values for the Sherman-Morrison formula
    long double *v = (long double*)malloc(size * sizeof(long double));
    long double *u = (long double*)malloc(size * sizeof(long double));
    
    for (int i = 0; i < size; i++) {
        v[i] = 0.0;
        u[i] = inverse->data[i][index] / inverse->data[index][index];
    }
    u[index] = 1.0 / inverse->data[index][index];
    
    // Compute the updated inverse matrix using the Sherman-Morrison formula
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i != index && j != index) {
                inverse->data[i][j] = inverse->data[i][j] - u[i] * inverse->data[index][j] * v[j];
            }
        }
    }
    
    for (int i = 0; i < size; i++) {
        if (i != index) {
            inverse->data[i][index] = -u[i] * v[index];
            inverse->data[index][i] = -u[index] * v[i];
        }
    }
    
    inverse->data[index][index] = u[index];
    
    // Update the inverse matrix with the new column vector
    for (int i = 0; i < size; i++) {
        inverse->data[i][index] += u[i] * columnVector[i];
    }
    
    free(u);
    free(v);
}

// Function to update TWO VECTOR the inverse matrix using Sherman-Morrison formula v 1.0
void updateInverseMatrix_TWO_VECTOR(long double **A_inv, long double **A, long double *rowVector, long double *colVector, int n) {
    long double factor1 = 0.0;
    long double factor2 = 0.0;
    long double *temp1 = (long double*)malloc(n * sizeof(long double));
    long double *temp2 = (long double*)malloc(n * sizeof(long double));

    // Calculate the factors: factor1 = 1 / (1 + colVector * A_inv * rowVector) and factor2 = -1 / (1 + rowVector * A_inv * colVector)
    for (int i = 0; i < n; i++) {
        temp1[i] = 0.0;
        temp2[i] = 0.0;
        for (int j = 0; j < n; j++) {
            temp1[i] += colVector[j] * A_inv[j][i];
            temp2[i] += rowVector[j] * A_inv[j][i];
        }
    }

    for (int i = 0; i < n; i++) {
        factor1 += temp1[i] * rowVector[i];
        factor2 += temp2[i] * colVector[i];
    }
    factor1 = 1.0 / (1.0 + factor1);
    factor2 = -1.0 / (1.0 + factor2);

    // Update the inverse matrix: A_inv = A_inv + factor1 * A_inv * rowVector * colVector * A_inv
    long double **temp3 = (long double**)malloc(n * sizeof(long double*));
    for (int i = 0; i < n; i++) {
        temp3[i] = (long double*)malloc(n * sizeof(long double));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp3[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                temp3[i][j] += A_inv[i][k] * rowVector[k] * colVector[j];
            }
            temp3[i][j] *= factor1;
            A_inv[i][j] += temp3[i][j];
        }
    }

    // Update the inverse matrix: A_inv = A_inv + factor2 * A_inv * colVector * rowVector * A_inv
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp3[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                temp3[i][j] += A_inv[i][k] * colVector[k] * rowVector[j];
            }
            temp3[i][j] *= factor2;
            A_inv[i][j] += temp3[i][j];
        }
    }

    // Free the temporary memory
    for (int i = 0; i < n; i++) {
        free(temp3[i]);
    }
    free(temp3);
    free(temp1);
    free(temp2);
}

// Function to update TWO VECTOR the inverse matrix using Sherman-Morrison formula v 2.0
void updateInverseMatrix_TWO_VECTOR2(long double **A_inv, long double **A, long double *u, long double *v, int n) {
    long double factor = 0.0;
    long double *temp1 = (long double*)malloc(n * sizeof(long double));
    long double *temp2 = (long double*)malloc(n * sizeof(long double));

    // Calculate the factors: factor = 1 / (1 + v * A_inv * u)
    for (int i = 0; i < n; i++) {
        temp1[i] = 0.0;
        temp2[i] = 0.0;
        for (int j = 0; j < n; j++) {
            temp1[i] += v[j] * A_inv[j][i];
            temp2[i] += A_inv[i][j] * u[j];
        }
    }

    for (int i = 0; i < n; i++) {
        factor += temp1[i] * u[i];
        factor += v[i] * temp2[i];
    }
    factor = 1.0 / (1.0 + factor);

    // Calculate A_inv = A_inv - (A_inv * u * v * A_inv) / (1 + v * A_inv * u)
    long double **temp3 = (long double**)malloc(n * sizeof(long double*));
    for (int i = 0; i < n; i++) {
        temp3[i] = (long double*)malloc(n * sizeof(long double));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp3[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                temp3[i][j] += A_inv[i][k] * u[k] * v[j];
            }
            temp3[i][j] *= factor;
            A_inv[i][j] -= temp3[i][j];
        }
    }

    // Free the temporary memory
    for (int i = 0; i < n; i++) {
        free(temp3[i]);
    }
    free(temp3);
    free(temp1);
    free(temp2);
}

//////////////////////////////////// NEW //////////////////////////////////////////////////////////
void updateInverseMatrix0(long double** invMatrix, long double* column, int size) {
    // Allocate memory for temporary matrices
    long double** tempMatrix = (long double**)malloc(size * sizeof(long double*));
    long double* tempVector = (long double*)malloc(size * sizeof(long double));
    long double tempValue = 0.0;

    // Compute the intermediate values required for the update
    for (int i = 0; i < size; i++) {
        tempMatrix[i] = (long double*)malloc(size * sizeof(long double));
        for (int j = 0; j < size; j++) {
            tempMatrix[i][j] = invMatrix[i][j] * column[j];
            tempValue += tempMatrix[i][j];
        }
        tempVector[i] = tempValue;
        tempValue = 0.0;
    }

    // Compute the Sherman-Morrison update
    long double denominator = 1.0 / (1.0 + tempVector[0]);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == 0) {
                invMatrix[i][j] -= (tempMatrix[i][j] * denominator);
            } else {
                invMatrix[i][j] -= ((tempMatrix[i][j] * tempVector[i-1]) / (1.0 + tempVector[0]));
            }
        }
    }

    // Free temporary matrices
    for (int i = 0; i < size; i++) {
        free(tempMatrix[i]);
    }
    free(tempMatrix);
    free(tempVector);
}

void updateInverseMatrix1(long double** invMatrix, int size, long double* vet) {
    int i, j;
    long double* u = (long double*)malloc(size * sizeof(long double));
    long double* v = (long double*)malloc(size * sizeof(long double));

    // Calculate u = invMatrix * vector
    for (i = 0; i < size; i++) {
        u[i] = 0.0;
        for (j = 0; j < size; j++) {
            u[i] += invMatrix[i][j] * vet[j];
        }
    }

    // Calculate v = vector * invMatrix
    long double denominator = 0.0;
    for (i = 0; i < size; i++) {
        v[i] = vet[i];
        denominator += u[i] * vet[i];
    }

    // Calculate the updated inverse matrix
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            invMatrix[i][j] -= (u[i] * v[j]) / denominator;
        }
    }

    free(u);
    free(v);
}

//void updateInverseMatrix2(long double A[MAX_SIZE][MAX_SIZE], long double invA[MAX_SIZE][MAX_SIZE], int size, int row, int col, long double newValue)
void updateInverseMatrix2(long double** A, long double** invA, int size, int row, int col, long double newValue) {
    // Compute the Sherman-Morrison correction factor
    long double factor = 1.0 / (newValue - A[row][col]);

    // Update the inverse matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == row && j == col) {
                // Update the (row, col) element using the Sherman-Morrison formula
                invA[i][j] += factor * invA[i][j] * A[row][col];
            } else {
                // Update other elements based on the Sherman-Morrison formula
                invA[i][j] -= factor * invA[i][col] * A[row][j];
                invA[i][j] -= factor * A[i][col] * invA[row][j];
            }
        }
    }

    // Add the new column and row to the original matrix
    A[row][col] = newValue;
    for (int i = 0; i < size; i++) {
        if (i != row) {
            A[i][col] = 0.0;
        }
        if (i != col) {
            A[row][i] = 0.0;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
// TODO: Function to read a matrix from a file
Matrix* readMatrixFromFile(int argc, const char* filename) {
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
            fscanf(file, "%Lf", &matrix->data[i][j]);
        }
    }

    fscanf(file, "%d", &size);  // Read the size of the inverse matrix
    // Read values for the inverse matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fscanf(file, "%d", &size);
        }
    }
    fscanf(file, "%d", &size);  // Read index and value to change in original matrix
    fscanf(file, "%d", &size);
    fscanf(file, "%d", &size);

    fclose(file);
    return matrix;
}

// TODO: ...
long double dotProduct(long double* u, long double* v, int size) {
    long double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += u[i] * v[i];
    }
    return result;
}

// TODO: Function to compute the inverse of a matrix using the Sherman-Morrison formula 1
Matrix* calculateInverse(Matrix* matrix, int rowIndex, int colIndex, long double newValue) {
    int size = matrix->size;
    Matrix* inverse = allocateMatrix(size);

    // Initialize the inverse matrix as the identity matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            inverse->data[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Perform Sherman-Morrison algorithm to update the inverse matrix
    long double* u = (long double*)malloc(size * sizeof(long double));
    long double* v = (long double*)malloc(size * sizeof(long double));

    // Calculate u and v vectors
    for (int i = 0; i < size; i++) {
        u[i] = matrix->data[i][colIndex];
        v[i] = matrix->data[rowIndex][i];
    }

    // Calculate the denominator term in the Sherman-Morrison formula
    long double denominator = 1.0 + dotProduct(u, v, size);

    // Calculate the numerator matrix
    Matrix* numeratorMatrix = allocateMatrix(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            numeratorMatrix->data[i][j] = -matrix->data[i][colIndex] * matrix->data[rowIndex][j];
        }
    }

    // Calculate the updated inverse matrix using the Sherman-Morrison formula
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            inverse->data[i][j] = inverse->data[i][j] + (numeratorMatrix->data[i][j] / denominator);
        }
    }

    freeMatrix(numeratorMatrix);
    free(u);
    free(v);

    return inverse;
}

// Function to calculate the inverse of a matrix using Sherman-Morrison formula
void inverse_matrix6(int n, long double **matrix, long double ** inv_matrix, int row, int col, double new_value) {
    int i, j;
    long double det = 0.0;
    //long double u[n][n], v[n][n], temp[n][n];
    long double **u, **v, **temp;
    u = my_allocateMatrix(n,n);
    v = my_allocateMatrix(n,n);
    temp = my_allocateMatrix(n,n);

    // Calculate the determinant of the matrix
    for (i = 0; i < n; i++) {
        long double sub_det = matrix[0][i];
        for (j = 1; j < n; j++) {
            sub_det = sub_det * matrix[j][((i + j) % n)];
        }
        det += sub_det;
    }
    //det = determinant;

    // Calculate the u and v vectors
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            u[i][j] = matrix[row][j] * inv_matrix[j][col];
            v[i][j] = inv_matrix[i][col] * matrix[row][j];
        }
    }

    // Update the inverse matrix using Sherman-Morrison formula
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            temp[i][j] = inv_matrix[i][j] + (u[i][j] / (1 - new_value * det)) - (v[i][j] / (1 - new_value * det));
        }
    }

    //printf("\nUpdated Inverse Matrix:\n");
    // for (i = 0; i < n; i++) {
    //     for (j = 0; j < n; j++) {
    //         printf("%Lf ", temp[i][j]);
    //     }
    //     printf("\n");
    // }

    copyMatrix(temp, inv_matrix, n, n);
    my_freeMatrix(u,n);
    my_freeMatrix(v,n);
    my_freeMatrix(temp,n);
}

// Function to update the inverse matrix using the Sherman-Morrison formula
void updateInverseMatrix7(int n, long double ** inv_matrix, long double * row, long double *column) {
    double u[n], v[n], x[n], y[n];
    int i, j;

    // Calculate u = inv_matrix * row
    for (i = 0; i < n; i++) {
        u[i] = 0.0;
        for (j = 0; j < n; j++) {
            u[i] += inv_matrix[i][j] * row[j];
        }
    }

    // Calculate v = column * inv_matrix
    for (i = 0; i < n; i++) {
        v[i] = 0.0;
        for (j = 0; j < n; j++) {
            v[i] += column[j] * inv_matrix[j][i];
        }
    }

    // Calculate denominator = 1 + v * u
    double denominator = 1.0;
    for (i = 0; i < n; i++) {
        denominator += v[i] * u[i];
    }

    // Calculate x = inv_matrix * u / denominator
    for (i = 0; i < n; i++) {
        x[i] = 0.0;
        for (j = 0; j < n; j++) {
            x[i] += inv_matrix[i][j] * u[j];
        }
        x[i] /= denominator;
    }

    // Calculate y = v / denominator
    for (i = 0; i < n; i++) {
        y[i] = v[i] / denominator;
    }

    // Update the inverse matrix using the Sherman-Morrison formula
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            inv_matrix[i][j] -= x[i] * y[j];
        }
    }
}

// Function to calculate the inverse of a matrix using the Sherman-Morrison formula
void inverse_matrix_8(int n, long double **matrix, long double ** inv_matrix) {
    int i, j, k;
    long double determinant = 0.0;
    long double temp[n][n];

    // Calculate the determinant of the matrix
    for (i = 0; i < n; i++) {
        long double sub_det = matrix[0][i];
        for (j = 1; j < n; j++) {
            sub_det = sub_det * matrix[j][(i + j) % n];
        }
        determinant += sub_det;
    }

    // Calculate the inverse matrix of the original matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            long double sub_det = 1.0;
            int row, col;
            for (row = 0; row < n; row++) {
                if (row != i) {
                    for (col = 0; col < n; col++) {
                        if (col != j) {
                            sub_det *= matrix[row][col];
                        }
                    }
                }
            }
            temp[i][j] = sub_det / determinant;
        }
    }

    // Apply the Sherman-Morrison formula to update the inverse matrix
    for (k = 0; k < n; k++) {
        long double u[n];
        for (i = 0; i < n; i++) {
            u[i] = temp[i][k];
        }
        long double v[n];
        for (i = 0; i < n; i++) {
            v[i] = matrix[i][k];
        }
        long double alpha = 0.0;
        for (i = 0; i < n; i++) {
            alpha += v[i] * inv_matrix[k][i];
        }
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if (i != k && j != k) {
                    inv_matrix[i][j] = temp[i][j] - (u[i] * v[j]) / (1 + alpha);
                }
            }
        }
        for (i = 0; i < n; i++) {
            if (i != k) {
                inv_matrix[i][k] = -u[i] / (1 + alpha);
            }
        }
        for (i = 0; i < n; i++) {
            if (i != k) {
                inv_matrix[k][i] = v[i] / (1 + alpha);
            }
        }
        inv_matrix[k][k] = 1 / (1 + alpha);
    }

    // Print the inverse matrix
    printf("\nInverse Matrix:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%Lf ", inv_matrix[i][j]);
        }
        printf("\n");
    }
}

/*
I apologize for the confusion. The Sherman-Morrison formula is typically used to update the inverse of a matrix when a new column or row is added or removed from the original matrix. It is not used to directly update a specific value in the inverse matrix.

If you want to update a specific value in the inverse matrix without altering the original matrix, you would typically use other techniques such as matrix operations or Gaussian elimination. The Sherman-Morrison formula is specifically designed for efficient updates when adding or removing columns or rows.

If you can provide more information about the specific context or problem you're trying to solve, I would be happy to assist you further.
*/

int main(int argc, char *argv[]) {
    Matrix *originalMatrix = NULL, *inverseMatrix = NULL;

    if (argc <=2){
        //read from keyborad or input file with two matrix and change command
        if (read_matrix_from_keyboard(&originalMatrix, &inverseMatrix) != -1){
            printf("Matriz Original %dx%d:\n",originalMatrix->size, originalMatrix->size);
            //printMatrix(originalMatrix);

            printf("\nMatrix Inversa %dx%d:\n",inverseMatrix->size, inverseMatrix->size);
            //printMatrix(inverseMatrix);
            
            long double new_var = originalMatrix->value_to_change;// 69<->99   2x1
            //long double line_vet[] = {12,  new_var,  68,  30,  83,  31,  63,  24,  68,  36};
            long double line_vet[] = {12,  new_var,  68,  0,  0,  0,  0,  0,  0,  0};

            long double _line_vet[] = {0.00236657, 0.00286759, 0.00390017, -0.000206773, -0.00919492, -0.000606505, 0.01009, 0.00296859, -0.000616597, -0.00655167};

            long double col_vet[] = {0,           
                                28,           
                                new_var,           
                                3,           
                                74,           
                                27,           
                                26,           
                                96,           
                                9,           
                                61};

            long double _col_vet[] = {-0.0161054,
                            0.00546049,  
                            0.00390017,  
                            0.00557017,  
                            0.0206132,   
                            0.00444797,  
                            -0.0108293,  
                            -0.00197925,
                            -0.00450104, 
                            -0.0119108};
            
            ///*
            if(atoi(argv[1])==-1){
                originalMatrix->data[ originalMatrix->i ][ originalMatrix->j ] = originalMatrix->value_to_change;//////////////// 99-69 ////////////////
                printf("%s\n", "Inverse Matrix by Gauss-Jordan elimination with row swaps");
                long double** temp = my_allocateMatrix(originalMatrix->size, originalMatrix->size);
                copyMatrix(originalMatrix->data, temp, originalMatrix->size, originalMatrix->size);
                long double** buffer = my_invertMatrix(temp, originalMatrix->size);
                my_freeMatrix(buffer, originalMatrix->size);
                my_freeMatrix(temp, originalMatrix->size);
            }
            //*/
            else{
                printf("\nNova Matrix Inversa %dx%d:\n",inverseMatrix->size, inverseMatrix->size);
                //printf("************ New Inverse Matrix (Sherman-Morrison):\n");
                if(atoi(argv[1])==0)
                    updateInverseMatrix_ONE_VECTOR(inverseMatrix, col_vet, originalMatrix->j);
                // updateInverseMatrix_TWO_VECTOR(inverseMatrix->data,originalMatrix->data, col_vet, line_vet, originalMatrix->size );
                
                else if(atoi(argv[1])==1)
                    updateInverse_ONE_VALUE(&originalMatrix, &inverseMatrix, originalMatrix->i , originalMatrix->j, originalMatrix->value_to_change);
                else if(atoi(argv[1])==2)
                    updateInverseMatrix_TWO_VECTOR2(inverseMatrix->data,originalMatrix->data, col_vet, line_vet, originalMatrix->size );
                else if(atoi(argv[1])==3)
                    updateInverseMatrix0(inverseMatrix->data, col_vet, originalMatrix->size);
                else if(atoi(argv[1])==4)
                    updateInverseMatrix1(inverseMatrix->data, originalMatrix->size, line_vet);
                else if(atoi(argv[1])==5)
                    updateInverseMatrix2(originalMatrix->data, inverseMatrix->data, originalMatrix->size, originalMatrix->i, originalMatrix->j, originalMatrix->value_to_change);
                else if(atoi(argv[1])==6)
                    inverse_matrix6(inverseMatrix->size, originalMatrix->data, inverseMatrix->data, inverseMatrix->i, inverseMatrix->j, inverseMatrix->value_to_change);
                else if(atoi(argv[1])==7)
                   updateInverseMatrix7(originalMatrix->size, inverseMatrix->data, line_vet, col_vet);
                else if(atoi(argv[1])==8)
                   inverse_matrix_8(originalMatrix->size, originalMatrix->data, inverseMatrix->data);

                printMatrix(inverseMatrix);
                //printf("%d%s\n", atoi(argv[1]), "-----------------------------------------------------------------------------------------------------------------------------------------------------\n");
            }
            //////////////////////////////////////DETERMINANTE MATRIZ//////////////////////////////////////////////////////
            ///*
            gsl_matrix *matrix = gsl_matrix_alloc(originalMatrix->size, originalMatrix->size);
            for (int i = 0; i < originalMatrix->size; i++)
                for (int j = 0; j < originalMatrix->size; j++)
                    gsl_matrix_set(matrix, i, j, (double) inverseMatrix->data[i][j] );

            // Compute the determinant
            gsl_permutation *perm = gsl_permutation_alloc(originalMatrix->size);
            int signum;
            double determinant;

            gsl_linalg_LU_decomp(matrix, perm, &signum);
            determinant = gsl_linalg_LU_det(matrix, signum);

            printf("Determinante: %lf\n", determinant);

            // Clean up allocated memory
            gsl_matrix_free(matrix);
            gsl_permutation_free(perm);

            //*/
            ////////////////////////////////////////////////////////////////////////////////////////////
        }
        else{
            printf("ALOCATION ERROR\n");
            return 1;
        }

    }
    else{
        //read from two files, first file has main matrix and another file has inversive matrix and change command    
        originalMatrix = readMatrixFromFile(argc, argv[2]); // "matrix.in"
        if (originalMatrix == NULL) {
            printf("Error reading the matrix from the file.\n");
            return 1;
        }

        inverseMatrix = readMatrixFromFile(argc, argv[3]); // "inverse_matrix.in"

        if (inverseMatrix == NULL) {
            printf("Error reading the inverse matrix from the file.\n");
            return 1;
        }

        //printf("Original Matrix:\n");
        //printMatrix(originalMatrix);

        //printf("\nInverse Matrix:\n");
        //printMatrix(inverseMatrix);

        // Set a new value in the original matrix
        int rowIndex, colIndex;
        long double newValue;
        //printf("\nEnter the row index, column index, and new value to set in the original matrix: ");
        //scanf("%d %d %Lf", &rowIndex, &colIndex, &newValue);
        rowIndex = 2;
        colIndex = 2;
        newValue = 68;

        if (rowIndex >= 0 && rowIndex < originalMatrix->size && colIndex >= 0 && colIndex < originalMatrix->size) {
            // Update the original matrix
            originalMatrix->data[rowIndex][colIndex] = newValue;

            // Calculate the updated inverse matrix using the Sherman-Morrison algorithm
            Matrix* updatedInverseMatrix = calculateInverse(originalMatrix, rowIndex, colIndex, newValue);

            printf("\nMatriz Original:\n");
            printMatrix(originalMatrix);

            printf("\nNova Matriz Inversa:\n");
            printMatrix(updatedInverseMatrix);

            freeMatrix(updatedInverseMatrix);
        } else {
            printf("Invalid row or column index.\n");
        }
    }

    freeMatrix(originalMatrix);
    freeMatrix(inverseMatrix);

    return 0;
}
