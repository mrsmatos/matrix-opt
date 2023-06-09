#include <stdio.h>
#include <stdlib.h>

// Function to update the inverse matrix using Sherman-Morrison formula
void updateInverseMatrix(int n, double A[][n], double invA[][n], double v[], double u[]) {
    double alpha, beta, gamma;
    double w[n][n];

    // Calculate the intermediate matrix w
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                w[i][j] = 1.0 / (v[i] - beta);
            else
                w[i][j] = 0.0;
        }
    }

    // Update the inverse matrix using Sherman-Morrison formula
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            invA[i][j] = invA[i][j] - (u[i] * w[j][i]);
        }
    }

    // Calculate the intermediate values alpha and beta
    alpha = 1.0 / (1.0 - (beta * u[0]));
    beta = u[0];

    // Update the inverse matrix using Sherman-Morrison formula
    for (int i = 0; i < n; i++) {
        invA[i][0] = alpha * invA[i][0];
    }

    // Update the inverse matrix using Sherman-Morrison formula
    for (int i = 1; i < n; i++) {
        gamma = invA[i][0];
        for (int j = 0; j < n; j++) {
            invA[j][i] = invA[j][i] + (gamma * u[j]);
        }
    }
}

// Function to print a matrix
void printMatrix(int n, double A[][n]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f\t", A[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int n = 3;  // Size of the matrix

    double A[][3] = {{1, 2, 3},
                     {4, 5, 6},
                     {7, 8, 9}};

    double invA[][3] = {{-0.67, 0.33, 0.00},
                        {0.33, -0.67, 0.33},
                        {0.00, 0.33, -0.67}};

    double v[] = {10, 11, 12};
    double u[] = {-1, 1, -1};

    printf("Original Matrix:\n");
    printMatrix(n, A);

    printf("\nOriginal Inverse Matrix:\n");
    printMatrix(n, invA);

    printf("\nRank-One Update Vector (v):\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f\t", v[i]);
    }

    printf("\nRank-One Update Vector (u):\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f\t", u[i]);
    }

    updateInverseMatrix(n, A, invA, v, u);

    printf("\nUpdated Inverse Matrix:\n");
    printMatrix(n, invA);

    return 0;
}

