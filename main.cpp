#include <iostream>
#include <omp.h>
#include <chrono>
using namespace std;

double * gauss(double **a, double *y, int n)
{
    double *x, max;
    int k, index;
    const double eps = 0.00001;
    x = new double[n];
    k = 0;
    while (k < n)
    {
        // Поиск строки с максимальным a[i][k]
        max = abs(a[k][k]);
        index = k;
        for (int i = k + 1; i < n; i++)
        {
            if (abs(a[i][k]) > max)
            {
                max = abs(a[i][k]);
                index = i;
            }
        }
        // Перестановка строк
        if (max < eps)
        {
            cout << "Решение получить невозможно из-за нулевого столбца ";
            cout << index << " матрицы A" << endl;
            return 0;
        }
        for (int j = 0; j < n; j++)
        {
            double temp = a[k][j];
            a[k][j] = a[index][j];
            a[index][j] = temp;
        }
        double temp = y[k];
        y[k] = y[index];
        y[index] = temp;
        // Нормализация уравнений
#pragma omp parallel
    {
#pragma omp for
        for (int i = k; i < n; i++)
        {
            double temp = a[i][k];
            if (abs(temp) < eps) continue; // для нулевого коэффициента пропустить
            for (int j = 0; j < n; j++)
                a[i][j] = a[i][j] / temp;
            y[i] = y[i] / temp;
            if (i == k)  continue; // уравнение не вычитать само из себя
            for (int j = 0; j < n; j++)
                a[i][j] = a[i][j] - a[k][j];
            y[i] = y[i] - y[k];
        }
        }
        k++;
    }

//    // обратная подстановка
#pragma omp parallel
    {
#pragma omp for
    for (k = n - 1; k >= 0; k--)
    {
        x[k] = y[k];
        for (int i = 0; i < k; i++)
            y[i] = y[i] - a[i][k] * x[k];
    }
}
    return x;
}

double random(const int min, const int max)
{
    if (min == max)
        return min;
    return min + rand() % (max - min);
}


int main()
{
    system("chcp 65001");
    double **a, *y, *x;
    int count_equation;

    for (int num_threads = 6; num_threads <= 8; num_threads += 1){
        omp_set_num_threads(num_threads);
        for (int i = 1; i<=8; i++) {
            count_equation = 500*i;
            omp_set_num_threads(num_threads);
            a = new double*[count_equation];
            y = new double[count_equation];
            for (int i = 0; i < count_equation; i++)
            {
                a[i] = new double[count_equation];
                for (int j = 0; j < count_equation; j++)
                {
                    a[i][j] = random(10, 100);
                }
                y[i] = random(10, 100);
            }

            auto start = std::chrono::high_resolution_clock::now();
            x = gauss(a, y, count_equation);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Время выполнения: " << duration.count() / 10 << " мс. " << "Потоки: " << num_threads <<" "
                      << "Уравнений: " << count_equation << std::endl;
        }
    }


    return 0;
}