// Samuel Kuchta (xkucht11)
// 5.4.2025


#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

const int MASTER_RANK = 0;
const char* INPUT_FILE = "numbers";
const int TAG = 0;
const MPI_Comm COMM = MPI_COMM_WORLD;

// build with -DDEBUG for printing synchronisation details
#ifdef DEBUG
    #define DEBUG_PRINT(...) do { \
        printf(__VA_ARGS__); \
    } while(0)
#else
    #define DEBUG_PRINT(...)
#endif

#define MPI_CHECK(call) { \
    int status = (call); \
    if (status != MPI_SUCCESS) { \
        cerr << "MPI Error in " << #call << endl; \
        MPI_Abort(COMM, EXIT_FAILURE); \
    } \
}

// comparison and swap logic
void compareWithPartner(int& number, int rank, int partner, bool is_left) {
    int received;
    MPI_CHECK(MPI_Sendrecv(&number, 1, MPI_INT, partner, TAG,
                          &received, 1, MPI_INT, partner, TAG,
                          COMM, MPI_STATUS_IGNORE));
      
    bool to_exhange = is_left ? number > received : number < received;

    if (to_exhange) {
        DEBUG_PRINT("[P%d] Taking value %d\n", rank, received);
        number = received;
    }

}

// Phase handling with correct (based on odd or even phase) partner selection
void handlePhase(int& number, int rank, int size, int phase) {
    int partner = -1;
    bool is_left = false;
    int phase_is_even = phase % 2;

    if (rank % 2 == phase_is_even && rank+1 < size) {
        partner = rank + 1;
        is_left = true;
    }
    else if (rank % 2 != phase_is_even && rank-1 >= 0) {
        partner = rank - 1;
        is_left = false;
    }

    if (partner != -1) {
        DEBUG_PRINT("[P%d] Partnering with [P%d] in phase %d\n", rank, partner, phase);
        compareWithPartner(number, rank, partner, is_left);
    }
}

void oddEvenTranspositionSort(int& number, int rank, int size) {
    for (int phase = 0; phase < size; ++phase) {
        handlePhase(number, rank, size, phase);
        MPI_CHECK(MPI_Barrier(COMM));
    }
}

vector<unsigned char> masterReadAndDistribute(int proc_count) {
    ifstream file(INPUT_FILE, ios::binary);
    if (!file) {
        cerr << "Error opening input file" << endl;
        MPI_CHECK(MPI_Abort(COMM, EXIT_FAILURE));
    }

    file.seekg(0, ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);

    vector<unsigned char> bytes(file_size);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), file_size)) {
        cerr << "Error reading input file" << endl;
        MPI_CHECK(MPI_Abort(COMM, EXIT_FAILURE));
    }

    if (bytes.size() != static_cast<size_t>(proc_count)) {
        cerr << "Input mismatch: " << bytes.size() 
             << " numbers vs " << proc_count << " processes" << endl;
        MPI_CHECK(MPI_Abort(COMM, EXIT_FAILURE));
    }

    DEBUG_PRINT("Input sequence:\n");
    for (size_t i = 0; i < bytes.size(); ++i) {
        cout << static_cast<int>(bytes[i]) 
             << (i < bytes.size()-1 ? " " : "\n");
    }

    for (int i = 1; i < proc_count; ++i) {
        int num = bytes[i];
        MPI_CHECK(MPI_Send(&num, 1, MPI_INT, i, TAG, COMM));
    }

    return bytes;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, size;
    MPI_CHECK(MPI_Comm_rank(COMM, &rank));
    MPI_CHECK(MPI_Comm_size(COMM, &size));

    int local_number = 0;
    if (rank == MASTER_RANK) {
        auto input = masterReadAndDistribute(size);
        local_number = input[MASTER_RANK];
    } else {
        MPI_CHECK(MPI_Recv(&local_number, 1, MPI_INT, MASTER_RANK, TAG, COMM, MPI_STATUS_IGNORE));
    }

    oddEvenTranspositionSort(local_number, rank, size);

    vector<int> sorted;
    if (rank == MASTER_RANK) {
        sorted.resize(size);
        sorted[0] = local_number;
        for (int i = 1; i < size; ++i) {
            MPI_CHECK(MPI_Recv(&sorted[i], 1, MPI_INT, i, TAG, COMM, MPI_STATUS_IGNORE));
        }
    } else {
        MPI_CHECK(MPI_Send(&local_number, 1, MPI_INT, MASTER_RANK, TAG, COMM));
    }

    if (rank == MASTER_RANK) {
        DEBUG_PRINT("\nSorted sequence:\n");
        for (int num : sorted) {
            cout << num << endl;
        }
    }

    MPI_CHECK(MPI_Finalize());
    return EXIT_SUCCESS;
}