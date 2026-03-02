#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

using namespace std;

// Глобальные параметры задачи
const int N = 1000;         // Размерность матрицы (по заданию Лаб 8)
const int GROUP_ID = 4;     // Параметр k из Лаб 2 (номер группы)
const int MAX_ITER = 1000;  // Количество итераций

// --- Функции генерации матрицы ---

vector<vector<double>> generateMatrix() {
    vector<vector<double>> A(N, vector<double>(N));
    
    
    
    mt19937 gen(123); // Фиксированный seed для воспроизводимости
    uniform_real_distribution<> dis(-100.0, 0.0);

    // 1. Заполнение недиагональных элементов и симметризация
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            double val = dis(gen);
            A[i][j] = val;
            A[j][i] = val; 
        }
    }

    // 2. Вычисление диагональных элементов (обеспечение диагональной преобладания)
    // Формула из кода Лаб 2: A[i][i] = -sum(row) + 10*k
    for (int i = 0; i < N; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            row_sum += A[i][j];
        }
        A[i][i] = -row_sum + 10.0 * GROUP_ID;
    }
    
    return A;
}

// --- Вспомогательные функции математики ---

// Умножение матрицы на вектор: v = A * u
vector<double> matVecMul(const vector<vector<double>>& A, const vector<double>& u) {
    vector<double> v(N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            v[i] += A[i][j] * u[j];
        }
    }
    return v;
}

// Евклидова норма
double normEuclid(const vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return sqrt(sum);
}

// Максимум-норма (возвращает значение и индекс)
pair<double, int> normMax(const vector<double>& v) {
    double maxVal = -1.0;
    int index = -1;
    for (int i = 0; i < N; ++i) {
        if (abs(v[i]) > maxVal) {
            maxVal = abs(v[i]);
            index = i;
        }
    }
    return {maxVal, index};
}

// Скалярное произведение
double dotProduct(const vector<double>& a, const vector<double>& b) {
    double res = 0.0;
    for (int i = 0; i < N; ++i) res += a[i] * b[i];
    return res;
}

// Вычитание векторов
vector<double> vecSub(const vector<double>& a, const vector<double>& b) {
    vector<double> res(N);
    for (int i = 0; i < N; ++i) res[i] = a[i] - b[i];
    return res;
}

// Умножение вектора на скаляр
vector<double> vecScale(const vector<double>& v, double s) {
    vector<double> res(N);
    for (int i = 0; i < N; ++i) res[i] = v[i] * s;
    return res;
}

// Деление вектора на скаляр
vector<double> vecDiv(const vector<double>& v, double s) {
    vector<double> res(N);
    for (int i = 0; i < N; ++i) res[i] = v[i] / s;
    return res;
}

// Знак числа
double sign(double x) {
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    return 0.0;
}

// --- Основные алгоритмы  ---

// Случай 1: Максимум-норма
void solveMaxNorm(const vector<vector<double>>& A) {
    cout << "\n=== Случай 1: Использование Максимум-нормы ===" << endl;
    
    vector<double> u(N, 1.0); // Начальное приближение
    
    // Нормировка начального вектора
    u = vecDiv(u, normMax(u).first);

    for (int k = 1; k <= MAX_ITER; ++k) {
        // v^{k+1} = A * u^k
        vector<double> v = matVecMul(A, u);

        // Норма ||v||_inf
        pair<double, int> maxInfo = normMax(v);
        double currentNorm = maxInfo.first;
        int maxIndex = maxInfo.second;

        // Lambda ~ v_i * sign(u_i)
        double lambda = v[maxIndex] * sign(u[maxIndex]);

        // Вывод результатов для k=999, 1000
        if (k == MAX_ITER - 1 || k == MAX_ITER) {
            // Вектор невязки для оценки точности
            vector<double> lambda_uk = vecScale(u, lambda);
            vector<double> residualVec = vecSub(v, lambda_uk);
            double residualNorm = normMax(residualVec).first;

            // u для вывода (нормированный v)
            vector<double> u_next = vecDiv(v, currentNorm);

            cout << "k = " << k << endl;
            cout << "  Lambda_1 = " << fixed << setprecision(6) << lambda << endl;
            cout << "  Норма невязки ||Av - lambda*u||_inf = " << scientific << residualNorm << endl;
            cout << "  5 первых координат u^k: ";
            cout << fixed << setprecision(4);
            for(int i=0; i<5; ++i) cout << u_next[i] << " ";
            cout << endl;
        }

        // Нормировка для следующего шага
        u = vecDiv(v, currentNorm);
    }
}

// Случай 2: Евклидова норма
void solveEuclideanNorm(const vector<vector<double>>& A) {
    cout << "\n=== Случай 2: Использование Евклидовой нормы ===" << endl;
    
    vector<double> u(N, 1.0);
    u = vecDiv(u, normEuclid(u)); // Нормировка

    for (int k = 1; k <= MAX_ITER; ++k) {
        vector<double> v = matVecMul(A, u);

        // Для симметричной матрицы Lambda ~ (v, u)
        double lambda = dotProduct(v, u);

        if (k == MAX_ITER - 1 || k == MAX_ITER) {
            vector<double> lambda_uk = vecScale(u, lambda);
            vector<double> residualVec = vecSub(v, lambda_uk);
            double residualNorm = normEuclid(residualVec);
            
            double vNorm = normEuclid(v);
            vector<double> u_next = vecDiv(v, vNorm);

            cout << "k = " << k << endl;
            cout << "  Lambda_1 = " << fixed << setprecision(6) << lambda << endl;
            cout << "  Норма невязки ||Av - lambda*u||_2 = " << scientific << residualNorm << endl;
            cout << "  5 первых координат u^k: ";
            cout << fixed << setprecision(4);
            for(int i=0; i<5; ++i) cout << u_next[i] << " ";
            cout << endl;
        }

        // Нормировка
        double vNorm = normEuclid(v);
        u = vecDiv(v, vNorm);
    }
}

int main() {
    setlocale(LC_ALL, "Russian");
    cout << "Лабораторная работа 8. Итерационный степенной метод." << endl;
    cout << "Размер матрицы n = " << N << ", Группа (параметр k) = " << GROUP_ID << endl;
    cout << "Генерация матрицы..." << endl;
    
    vector<vector<double>> A = generateMatrix();
    
    solveMaxNorm(A);
    solveEuclideanNorm(A);

    return 0;
}