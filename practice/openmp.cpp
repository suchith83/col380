#include <omp.h>
#include <vector>

using namespace std;

int main() { 
    int n = 4;
    
    // the function omp_get_thread_num() returns the thread number
    int x= 0;
    #pragma omp parallel num_threads(n)
    {
        int id = omp_get_thread_num();
        #pragma omp atomic
            x++;
        // this block of code is executed by each of n threads
    }

    int i, k, l;
    float j;
    #pragma omp parallel num_threads(n) shared(i, j) private(k, l)
    {
        int m, n;
        #pragma omp flush (i, j, j)
    }

    // peterson's algorithm
    // initialize boolean array flag with false
    vector<bool> flag(2, false);
    int defer = -1;
    #pragma omp parallel num_threads(2)
    {
        int myId = omp_get_thread_num();
        flag[myId] = true;
        #pragma omp flush(flag, defer)
        defer = myId;
    }
    int N = 10;
    omp_lock_t lock[N];
    for (int i = 0; i < N; i++) omp_init_lock(&lock[i]);
    #pragma omp parallel
        for(int item=0; item<N; item++) {
            if(omp_test_lock(&lock[item])) {
                // critical section
                omp_unset_lock(&lock[item]);
            } // otherwise some other thread has this lock, move on
        }
    for (int i = 0; i < N; i++) omp_destroy_lock(&lock[i]);


    // sharing loops work 
    int num = N/n;
    #pragma omp parallel for num_threads(n)
    {
        int st = num * omp_get_thread_num();
        for(int item=st; item < st+num; item++) {
            // work on item
        }
    }

    #pragma omp task
    {
        //code block
    }

    return 0;

}