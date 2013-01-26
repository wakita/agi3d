#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

typedef boost::numeric::ublas::matrix<float> fmatrix;
typedef boost::numeric::ublas::vector<float> fvector; 

using namespace std;

extern "C" int ssyev_(const char *jobz, const char
        *uplo, int *n, float *a, int *lda, float *w,
        float *work, int *lwork, int *info);

float * solver(float *, float *, float, float *);

//Params about Graph
int N;
int M;
int ** D;
vector<int> *neighbor;
vector< pair<int, int> > edges;

//Graph Layouts
int dim;
fmatrix P; //high dimensional layout
fmatrix E; //Unit vectors
fmatrix C; //3D Layout

static float init[15];

int strToInt(string &str){
    int t; stringstream ss;
    ss << str; ss >> t;
    return t;
}

void loadData(){

    ifstream ifs("data/jazzDisMat.txt");
    istringstream ist;
    string str;
    if(ifs.fail()){
        cerr << "File not found\n";
        exit(0);
    }

    getline(ifs,str);
    vector<string> tmp;
    boost::algorithm::split(tmp,str,boost::algorithm::is_space());
    N = tmp.size();

    D = new int*[N];
    neighbor = new vector<int>[N];
    for(int i = 0; i < N; i++){
        D[i] = new int[N];
        D[0][i] = strToInt(tmp[i]);
    }

    int j = 1;
    while(getline(ifs,str)){
        boost::algorithm::split(tmp,str,boost::algorithm::is_space());
        for(int i = 0; i < N; i++){
            D[j][i] = strToInt(tmp[i]);
        }
        j++;
    }

    for(int i = 0; i < N; i++){
        for(int j = i+1; j < N; j++){
            if(D[i][j] == 1){
                neighbor[i].push_back(j);
                neighbor[j].push_back(i);
                edges.push_back(make_pair(i,j));
            }
        }
    }
    M = edges.size();
}

void printmat(int _N, int _M, float *A, int LDA) {
    float mtmp;
    printf("[");
    for (int i = 0; i < _N; i++) {
        printf("[");
        for (int j = 0; j < _M; j++) {
            mtmp = A[i + j * LDA];
            printf("%5f", mtmp);
            if (j < _M - 1) printf(", ");
        }
        if (i < _N - 1) printf("];\n");
        else printf("]");
    }
    printf("]");
}

void calcInitLayout(){
    //MDS
    fmatrix D2(N,N);
    for(int i = 0; i < N; i ++){
        for(int j = i; j < N; j++){
            float tmp = (float)(D[i][j]*D[i][j]);
            D2(i,j) = tmp; D2(j,i) = tmp; 
        }
    }

    boost::numeric::ublas::identity_matrix<float> I(N,N);
    boost::numeric::ublas::scalar_matrix<float> S(N,N,(float)1.0/N);
    fmatrix H(N,N);
    H = I - S;
    fmatrix b(N,N); b = prod(H,D2);
    fmatrix c(N,N); c = prod(b,H);
    fmatrix B(N,N); B = -0.5*c; 

    int lwork, info;
    float *A = new float[N*N];
    float *w = new float[N];

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            A[i+j*N] = B(i,j);
        }
    }
    lwork = -1;
    float *work = new float[1];
    ssyev_("V", "U", &N, A, &N, w, work, &lwork, &info);
    lwork = (int)work[0];
    delete[]work;

    work = new float[max((int) 1, lwork)];
    ssyev_("V", "U", &N, A, &N, w, work, &lwork, &info);

    //printf("#eigenvalues \n"); printf("w ="); printmat(N, 1, w, 1); printf("\n");
    //printf("#eigenvecs \n"); printf("U ="); printmat(N, N, A, N); printf("\n");

    dim = 0;
    float eps = 0.01;
    for(int i = N-1; i >= 0; i--){
        if(w[i] > eps) dim++;
        else break;
    }

    if(dim > 50) dim = 50;

    P = fmatrix(N,dim);
    fmatrix L = fmatrix(N,dim);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < dim; j++){
            if( i==j ) L(i,i) = (float)sqrt(w[N-1-j]);
            else L(i,j) = 0.0f;
        }
    }

    fmatrix U(N,N);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            U(i,j) = A[j+(N-1-i)*N];
        }
    }

    //High Dimensional Layout : P
    P = prod(trans(U),L);

    fvector e1(dim), e2(dim), e3(dim);
    fvector e[3];

    for(int i = 0; i < 3; i++){
        e[i] = fvector(dim);
        for(int j = 0; j < dim; j++){
            if(j % 3 == i) e[i][j] = sqrt(w[N-1-j]);
            else e[i][j] = 0.0f;
        }
        e[i] = e[i] / norm_2(e[i]);
    }

    E = fmatrix(dim,3);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < dim; j++){
            E(j,i) = e[i][j];
        }
    }

    C = prod(P,E);
    //cout << "Initial Layout caluclation is done." << endl;
    delete[]work; delete[]w; delete[]A;

    init[0] = 0.9; init[1] = -0.1; init[2] = 0.1; 
    init[3] = -0.1; init[4] = 0.9; init[5] = 0.1;
    init[6] = 0.1; init[7] = 0.1; init[8] = 0.9;
    init[9] = 0.1; init[10] = -0.1; init[11] = 0.1;
    init[12] = -0.1; init[13] = 0.1; init[14] = 0.1;
}

int reprojection(int id, float pre_x, float pre_y, float pre_z, float new_x, float new_y, float new_z){
    float _pre[3]; float _new[3];
    _pre[0] = pre_x; _pre[1] = pre_y; _pre[2] = pre_z;
    _new[0] = new_x; _new[1] = new_y; _new[2] = new_z;

    fvector p(dim);
    float p_norm = 0, new_norm = 0, pre_norm = 0;

    for(int i = 0; i < dim; i++){
        p(i) = P(id,i);
        p_norm += p(i)*p(i);
    }

    for(int i = 0; i < 3; i++){
        new_norm += _new[i]*_new[i];
        pre_norm += _pre[i]*_pre[i];
    }

    if(new_norm < p_norm*0.85f && pre_norm < p_norm*0.85f){
        //cout << "pre:" << pre_x << " " << pre_y << " " << pre_z << endl;
        //cout << "new:" << new_x << " " << new_y << " " << new_z << endl;        
        /*
           float t = 0.001;

           if(new_norm > (1-t)*p_norm){
           for(int i = 0; i < 3; i++){
           _new[i] *= (1-t)*p_norm/new_norm;
           }
           }
           */

        fvector e0(dim), f0(dim), e1(dim), e2(dim), e3(dim);

        for(int i = 0; i < dim; i++){
            e1(i) = E(i,0);
            e2(i) = E(i,1);
            e3(i) = E(i,2);
        }

        f0 = p - _pre[0]*e1 - _pre[1]*e2 - _pre[2]*e3;
        float norm_f0 = norm_2(f0);
        e0 = f0 / norm_f0;
        //cout << inner_prod(e0,e1) << " " << inner_prod(e0,e2) << " " << inner_prod(e0,e3) <<  endl;

        //cout << norm_f0 << endl;

        float * res = solver(_pre, _new, norm_f0, init);

        if(res == NULL){
            init[0] = 0.9; init[1] = -0.1; init[2] = 0.1; 
            init[3] = -0.1; init[4] = 0.9; init[5] = 0.1;
            init[6] = 0.1; init[7] = 0.1; init[8] = 0.9;
            init[9] = 0.4; init[10] = -0.4; init[11] = 0.4;
            init[12] = -0.4; init[13] = 0.4; init[14] = 0.4;
            cout << "NULL" << endl;
            return 0;
        }

        fvector e1_new(dim), e2_new(dim), e3_new(dim), r1(dim), r2(dim);

        float a10 = (new_x - _pre[0]*res[0] - _pre[1]*res[1] - _pre[2]*res[2])/norm_f0;
        float a20 = (new_y - _pre[0]*res[3] - _pre[1]*res[4] - _pre[2]*res[5])/norm_f0;
        float a30 = (new_z - _pre[0]*res[6] - _pre[1]*res[7] - _pre[2]*res[8])/norm_f0;

        e1_new = a10*e0 + res[0]*e1 + res[1]*e2 + res[2]*e3;
        e2_new = a20*e0 + res[3]*e1 + res[4]*e2 + res[5]*e3;
        e3_new = a30*e0 + res[6]*e1 + res[7]*e2 + res[8]*e3;

        //init
        /*
        for(int i = 0; i < 15; i++){
            init[i] = res[i];
        }
        */

        r1 = res[9]*e1 + res[10]*e2 + res[11]*e3;
        r2 = res[12]*e1 + res[13]*e2 + res[14]*e3;

        float x = inner_prod(p,e1_new), y = inner_prod(p,e2_new), z = inner_prod(p,e3_new);
        float n_1 = norm_2(e1_new), n_2 = norm_2(e2_new), n_3 = norm_2(e3_new),
              n_r1 = norm_2(r1), n_r2 = norm_2(r2);

        float er1 = inner_prod(e1,r1) - inner_prod(e1_new,r1);
        float er2 = inner_prod(e1,r2) - inner_prod(e1_new,r2);
        float er3 = inner_prod(e2,r1) - inner_prod(e2_new,r1);
        float er4 = inner_prod(e2,r2) - inner_prod(e2_new,r2);
        float er5 = inner_prod(e3,r1) - inner_prod(e3_new,r1);
        float er6 = inner_prod(e3,r2) - inner_prod(e3_new,r2);

        float e1_e2 = inner_prod(e1_new,e2_new),
              e2_e3 = inner_prod(e2_new,e3_new),
              e1_e3 = inner_prod(e1_new,e3_new),
              r1_r2 = inner_prod(r1,r2);

        //cout << "== ERRORS ==" << endl;
        float errors1 = 0.0f;
        errors1 += (x - _new[0])*(x - _new[0]);
        errors1 += (y - _new[1])*(y - _new[1]);
        errors1 += (z - _new[2])*(z - _new[2]);

        float errors2 = 0.0f;
        errors2 += (n_1 - 1)*(n_1 - 1);
        errors2 += (n_2 - 1)*(n_2 - 1);
        errors2 += (n_3 - 1)*(n_3 - 1);
        errors2 += (n_r2 - 1 )*(n_r2 - 1);

        float errors3 = 0.0f;
        errors3 += (e1_e2)*(e1_e2);
        errors3 += (e2_e3)*(e2_e3);
        errors3 += (e1_e3)*(e1_e3);
        errors3 += (r1_r2)*(r1_r2);

        float errors4 = 0.0f;
        errors4 += er1*er1;
        errors4 += er2*er2;
        errors4 += er3*er3;
        errors4 += er4*er4;
        errors4 += er5*er5;
        errors4 += er6*er6;

        float change = inner_prod(e1,e1_new) + inner_prod(e2,e2_new) + inner_prod(e3,e3_new);
        if(change < 2.8) return 0;

        //cout << errors1 << endl;
        //cout << errors2 << endl;
        //cout << errors3 << endl;
        //cout << errors4 << endl;
        cout << sqrt(errors1+errors2+errors3+errors4) << endl;

        //        e1_new = e1_new / norm_2(e1_new);
        //        e2_new = e2_new / norm_2(e2_new);
        //        e3_new = e3_new / norm_2(e3_new);

        //Gram-Schmidt orthonormalization
        e1_new = e1_new / norm_2(e1_new);
        e2_new = e2_new - inner_prod(e1_new,e2_new)*e1_new; 
        e2_new = e2_new / norm_2(e2_new);
        e3_new = e3_new - inner_prod(e1_new,e3_new)*e1_new - inner_prod(e2_new,e3_new)*e2_new;;
        e3_new = e3_new / norm_2(e3_new);
        
        //cout << norm_2(e1_new) << " " << norm_2(e2_new) << " " << norm_2(e3_new) << endl;
        /*
           cout << "== debug ==" << endl;
           cout << x - _new[0] << endl;
           cout << y - _new[1] << endl;
           cout << z - _new[2] << endl;
           cout << n_1 - 1 << endl; 
           cout << n_2 - 1<< endl;
           cout << n_3 - 1<< endl;
           cout << n_r1 - 1 << endl;
           cout << n_r2 - 1 << endl;
           cout << e1_e2 << endl;
           cout << e2_e3 << endl;
           cout << e1_e3 << endl;
           cout << r1_r2 << endl;
           */

        for(int i = 0; i < dim; i++){
            E(i,0) = e1_new(i);
            E(i,1) = e2_new(i);
            E(i,2) = e3_new(i);
        }
        C = prod(P,E);
        return 1;
    }
    else{
        return 0;
    }
}

