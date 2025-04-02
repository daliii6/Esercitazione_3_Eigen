#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

// funzione per risolvere Ax = b usando la decomposizione PALU
VectorXd SolvePALU(const MatrixXd& A, const VectorXd& b) {
    PartialPivLU<MatrixXd> lu(A);
    return lu.solve(b);
}

// funzione per risolvere Ax = b usando la decomposizione QR
VectorXd SolveQR(const MatrixXd& A, const VectorXd& b) {
    HouseholderQR<MatrixXd> qr(A);
    return qr.solve(b);
}

// funzione per calcolare gli errori relativi
double Relerr(const VectorXd& x_es, const VectorXd& x_approx) {
    return (x_es - x_approx).norm() / x_es.norm();
}

int main() {
	// Definizione di un vettore di matrici contenente i coefficienti A dei tre sistemi lineari da risolvere
    vector<MatrixXd> A_matrici = {
        (MatrixXd(2,2) << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01).finished(),
        (MatrixXd(2,2) << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01).finished(),
        (MatrixXd(2,2) << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01).finished()
    };
    
	// Definizione di un vettore di vettori contenente i termini noti b dei tre sistemi lineari
    vector<VectorXd> b_vectors = {
        (VectorXd(2) << -5.169911863249772e-01, 1.672384680188350e-01).finished(),
        (VectorXd(2) << -6.394645785530173e-04, 4.259549612877223e-04).finished(),
        (VectorXd(2) << -6.400391328043042e-10, 4.266924591433963e-10).finished()
    };
    
	
	VectorXd x_es = VectorXd::Constant(2, -1.0);
	
    for (size_t i = 0; i < A_matrici.size(); i++) {
        MatrixXd A = A_matrici[i];
        VectorXd b = b_vectors[i];
        
        VectorXd x_palu = SolvePALU(A, b);
        VectorXd x_qr = SolveQR(A, b);
        
        double err_palu = Relerr(x_es, x_palu);
        double err_qr = Relerr(x_es, x_qr);

        cout << "Sistema " << i+1 << ":\n";
        cout << "  Soluzione con PALU: " << x_palu.transpose() << "\n";
        cout << "  Errore con PALU: " << err_palu << "\n";
        cout << "  Soluzione con QR: " << x_qr.transpose() << "\n";
        cout << "  Errore con QR: " << err_qr << "\n";
    }

    return 0;
}
