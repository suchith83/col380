#include <iostream>

using namespace std;

int main () {
    #pragma omp parallel
    {
        cout << "Hello, World!" << endl;
        int cnt = 0;
        int i = 0;
        #pragma omp for
        for (i = 0; i < 10; i++) {
            cnt++;
        }
        cout << cnt << endl;
    }
    
    return 0;
}