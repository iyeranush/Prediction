#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <list>
#include <vector>

/*
This file consists of 2 parts:
1. Simple Moving Average Filter
-Improved Simple Moving Average Filter. with less computations.
2. KalMan Filter

Input Data: 
-Positions of Reports, basically x,y coordinates.
-contains timestamps. Input in matrix "mat".
*/

using namespace std;
typedef vector<double> Row;
typedef vector<Row> Data;



class Kalman
{
 
    double T;
   
    //%Since we need to find the covariance for 2 columns,
    //%covariance matrix will be 2*2.
    public:
    
    Data Sum(const Data &mat)
    {
        //This is a function to calculate the sum of 
        //the elements in one column of a matrix.    
        //The sum gets calculated in sum_mat matrix.
        size_t rows= mat.size();
        size_t cols=mat[0].size();
        Data sum_mat(1,Row(2));
        for(size_t i=0;i<rows;i++)
        {
            for(int j=0;j<2;j++)
            {
                sum_mat[0][j]+= mat[i][j];
            }
        }
        
        return sum_mat;

    }
    Data mean(const Data &mat )
    {
        //This function subtracts mean from data. 
        size_t rows= mat.size();
        size_t cols=mat[0].size();
        Data sum_mat;
        //Calls Sum on the matrix mat.
        sum_mat= Sum(mat);
       
        Data mean_mat(rows,Row(2));
        double meanY=sum_mat[0][0]/rows;
        double meanX=sum_mat[0][1]/rows;
        for(size_t i=0;i<rows;i++)
        {
            mean_mat[i][0]=mat[i][0]-meanY;
            mean_mat[i][1]=mat[i][1]-meanX;
        }
       
        return mean_mat;
    }

    Data transpose(const Data &mat)
    {
        //This function calculates the transpose of a matrix.
        //Retruns the transpose of the matrix
        int rows= mat.size();
        int cols=mat[0].size();
        Data transpose_mat(cols,Row(rows));
        for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
        {
            transpose_mat[j][i]=mat[i][j];
        }
        return transpose_mat;

    }
    Data sum_matrix(const Data& A,const Data& B)
    {
        //This function calculates the sum of two matrix. and returns the resultant matrix
        size_t n=A.size();
        size_t m=A[0].size();
        Data D(n,Row(m,0));
                for(int i=0;i<n;i++)
                {
                    for(int j=0;j<m;j++)
                    {
                       
                        D[i][j]=A[i][j]+B[i][j];
                    }
                }
           
        return D;

    }
    Data subtract_matrix(const Data& A,const Data& B)
    {
        //This function calculates the subtraction of two matrix. and returns the resultant matrix
        size_t n=A.size();
        size_t m=A[0].size();
        Data D(n,Row(m,0));
        for(int i=0;i<n;i++)
                {
                    for(int j=0;j<m;j++)
                    {
                       
                        D[i][j]=A[i][j]-B[i][j];
                    }
                }

        return D;

    }
    Data multiply_matrix(const Data& A,const Data& B)
    {
    //This function calculates the multiplication of two matrix. and returns the resultant matrix
    size_t n=A.size();
    size_t m=A[0].size();
    size_t p=B[0].size();
    
    Data D(n,Row(p,0));
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<p;j++)
        {
            for(int k=0;k<m;k++)
            {
                D[i][j]+=A[i][k]*B[k][j];
            }
        }
    }
    return D;
    }

    Data deviation(const Data &mean_mat, size_t n)
    {
        //This function calculates the standard deviation of a matrix and returns the resultant matrix
        Data transpose_mat= transpose(mean_mat);
        //Dimension of deviation matrix is column*column of mean_mat
        size_t cols= mean_mat[0].size();
        Data deviation_mat(cols,Row(cols));
        deviation_mat=multiply_matrix(transpose_mat,mean_mat);
        return deviation_mat;
    }
    Data variance(const Data &deviation_mat, size_t n)
    {
        //Calculates the variance-covariance matrix. 
        //It is (deviation /total observations).
        size_t rows= deviation_mat.size();
        size_t cols= deviation_mat[0].size();
        double dev;
        Data covar(rows,Row(cols));

        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
               { dev=deviation_mat[i][j];
                covar[i][j]=dev/n;
            }
         }   
       
        return covar;

    }
    Data covariance(const Data &mat)
    {
        //Calculating covariance of observation mat
        //1.Mean
        //2.standard deviation
        //3.Variance
       
        size_t rows= mat.size();
        Data mean_mat=mean(mat);
       
        //Deviation calls mat'.mat
        Data deviation_mat;
        deviation_mat=deviation(mean_mat, rows);
       
        //variace does deviation/n;
        Data var_covar;
        var_covar= variance(deviation_mat,rows);
        
        return var_covar;
    }
    Data getSystemMatrix(double T, size_t n)
    {
        //System matrix is formed with identity matrix and T values.
        Data A(n,Row(n));
        for(size_t i=0;i<n;i++)
            A[i][i]=1;
        A[0][1]=T;
        A[n-2][n-1]=T;
        return A;

    }
    Data getMeasurementMatrix(size_t p,size_t n)
    {
        //Basic Measurement matrix is C=[1 0 0 0; 0 1 0 0]
        Data C(p,Row(n));
        C[0][0]=1;
        C[1][2]=1;
        return C;
    }
    Data getIdentity(size_t n)
    {
        //Function to form identity matrix of size n*n
        Data I(n,Row(n));
        for(size_t i=0;i<n;i++)
        {
            I[i][i]=1;
        }
        return I;
    }
    
    Data getIdentity(size_t n, double m)
    {
        //function overloading of getIdentity function.
         //Function to form identity matrix of size n*n and value at diogonal as m.
        Data I(n,Row(n));
        for(size_t i=0;i<n;i++)
        {
            I[i][i]=m;
        }
        return I;
    }
    Data getCovarianceNoise(size_t n)
    {
        //Assumption: Covariance noise is identity matrix 
        Data Q(n, Row(n));
        Q=getIdentity(n);
        return Q;
    }
     int getMinor(const Data &src,Data &dest, int row, int col, int n)
    {
    // indicate which col and row is being copied to dest
        //Calculating coFactors for finidng inverse of matrix
        int colCount=0,rowCount=0;
 
        for(int i = 0; i < n; i++ )
        {
        if( i != row )//Sub matrix.
        {
            colCount = 0;
            for(int j = 0; j < n; j++ )
            {
                // when j is not the element
                if( j != col )
                {
                    dest[rowCount][colCount] = src[i][j];
                    colCount++;
                }
            }
            rowCount++;
        }
        }
 
        return 1;
    }
    double determinant(const Data &mat, int n)
    {
   
        // stop the recursion when matrix is a single element
        if( n == 1 )
        return mat[0][0];
        // the determinant value
        double det = 0;
 
        // allocatting the cofactor matrix
        Data min(n-1,Row(n-1));
        for(int i = 0; i < n; i++ )
        {
        // get minor of element (0,i)
        getMinor( mat, min, 0, i , n);
        // the recusion is here.
        det += (i%2==1?-1.0:1.0) * mat[0][i] * determinant(min,n-1);
        }
      
        return det;
    }
   
    Data inverse_matrix(const Data &I)
    {
        //Inverse of matrix has two parts Determinats and cofactors of minors.
       // get the determinant of a
        size_t n=I.size();
        Data Inverse_mat(n,Row(n));
        double det = 1.0/determinant(I,n);
 
        // memory allocation
        Data min(n-1,Row(n-1));
 
        for(size_t j=0;j<n;j++)
        {
            for(size_t i=0;i<n;i++)
            {
            // get the co-factor (matrix) of A(j,i)
            getMinor(I,min,j,i,n);
            Inverse_mat[i][j] = det*determinant(min,n-1);
            if( (i+j)%2 == 1)
                Inverse_mat[i][j] = -Inverse_mat[i][j];
            }
        }
       
        return Inverse_mat;

    }
    void print(const Data &A)
    {
        size_t rows= A.size();
        size_t cols=A[0].size();
       
        for(size_t i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                cout<< A[i][j]<<"\t";
            }
            cout<<endl;
        }
        cout<<endl;

    }
    Data kalman_filter(const Data &mat, Data &A, Data &C, Data &Q, Data &R,  Data &X0, Data &P0 )
    {
        //Number of samples
        size_t rows=mat.size();
        size_t cols=mat[0].size()-1;
        size_t n= A.size();//system Order
        Data Xpredicted(rows+1,Row(n));//Kalman Predicted states here 311*4
        Data Xfiltered(rows+1,Row(n,0));//Kalman Filtered states here 311*4
      
        Xfiltered.at(0)=X0.at(0);//Filter Initialization.. c  PERFECT
       // cout<< Xfiltered[0][0]<<"\t"<<Xfiltered[0][1]<<endl;
        Data P(n,Row(n));
        P=P0;//Time=0... Assigned identity matrix with diagonal =100 PERFECT
        
        //Initial covariance matrix with uncertainity
        //Below are some dummy vectors too store temporary data.
        //Xpred= A*Xfilt
        Data tempPredTrans(1, Row(4));
        Data tempMat(2,Row(1));
       
        Data tempPred(4, Row(1));
        Data tempFilt(4, Row(1));
        tempFilt=transpose(X0);
        
        Data temp(4, Row(1));
        temp=multiply_matrix(A,tempFilt);
       
        Xpredicted.at(0)=transpose(temp).at(0);

        size_t crow=C.size();
        size_t ccol=C[0].size();
        Data Ctranspose(ccol,Row(crow));
        Ctranspose=transpose(C);
        //Uncertainty update  P = A*P*A'+Q
        P=sum_matrix(multiply_matrix(multiply_matrix(A,P),transpose(A)),Q);
       
        Data PCtranspose(n,Row(crow));
        Data CPCtranspose(crow,Row(crow));
        Data K(n,Row(crow));
        Data InverseCPCR(crow, Row(crow)),CPCR(crow, Row(crow));
       
        //Kalman filter iterations
        Data L, M, N;
        for(size_t t=1; t<rows+1;t++)
        {
            
            //To find K=P*C'*inv(C*P*C'+R)
            PCtranspose=multiply_matrix(P,Ctranspose);
            
            CPCtranspose=multiply_matrix(C,PCtranspose);
            
            CPCR= sum_matrix(CPCtranspose,R);
            
            InverseCPCR= inverse_matrix(CPCR);
           
            K=multiply_matrix(PCtranspose,InverseCPCR);
           
            //Filter update based on measurement mat
            tempPredTrans.at(0)= Xpredicted.at(t-1);

            tempMat[0][0]=mat[t-1][0];
            tempMat[1][0]=mat[t-1][1];
            //cout<< tempMat[0][0]<<"\t"<<tempMat[1][0]<<endl;
            tempPred=transpose(tempPredTrans);//4*1

            L=multiply_matrix(C,tempPred);/////CORRECT
            
            //cout<<"L:"<< L[0][0]<<"\t"<<L[1][0]<<endl;
            M=subtract_matrix(tempMat,L);

            //cout<<"M:"<< M[0][0]<<"\t"<<M[1][0]<<endl;
            N=multiply_matrix(K, M);

            //cout<<"N:"<< N[0][0]<<"\t"<<N[1][0]<<endl;
            tempFilt=sum_matrix(tempPred, N);

            //cout<< tempFilt[0][t]<<"\t"<<tempFilt[1][t]<<endl;
            Xfiltered.at(t) =transpose(tempFilt).at(0);

            //Uncertainity update
            P=subtract_matrix(P,multiply_matrix(K, multiply_matrix(C,P)));

            //Prediction
            tempPred=multiply_matrix(A,tempFilt);

            Xpredicted.at(t)=transpose(tempPred).at(0);

            //cout<< Xpredicted[t][0]<<"\t"<<Xpredicted[t][1]<<endl;
            P=sum_matrix(multiply_matrix(multiply_matrix(A,P),transpose(A)),Q);


            //Update A for new entry with timestamp
            if(t<rows)
                T=mat[t][2]-mat[t-1][2];
            else
                T=0.5;  
            A=getSystemMatrix(T, n);
            
            //cout<< "T"<<T<<endl;
            //cout<< Xfiltered[t][0]<<"\t"<<Xfiltered[t][1]<<endl;

        }
        return Xfiltered;


    }

    Data kalman_init(const Data &mat)
    {
       
       //Main components of kalman Filter:
        //Assumption: Velocity is constant.
        //We also have time interval between 2 records. Which we calculate in T to update A in the loop.
        //Initially T= 1st record timestamp
        T= mat[0][2];
        Data covar;
        //Covariance matrix of mat
        covar=covariance(mat);
      
        //Explain why n=4
        size_t n=4;
        Data A(n,Row(n));
        A=getSystemMatrix(T,n);
       //Get system Matrix of size 4*4
        size_t p=2;
        // Measurement matrix of size p × n
        Data C(p,Row(n));
        C=getMeasurementMatrix(p,n);
      //Covariance matrix of the process noise, n × n
        Data Q(n,Row(n));
        Q=getCovarianceNoise(n);
       
        //State 0 estimated X0 n*1 
        Data X0(1,Row(n));
       
        //Error covariance for X0, n*n
        Data P0(n,Row(n));
        P0=getIdentity(n,100);
        
        //Set r=10. Can try for 0,10,100,1000.. Selected 10 after experimentation for this data.
        Data R(p,Row(p));
        R=getIdentity(p,10);
        
        size_t rows= mat.size();
        size_t cols= mat[0].size();
        Data dataFiltered(rows+1,Row(cols-1));
       //apply kalman filter on the data.
        dataFiltered= kalman_filter(mat, A, C, Q,R, X0, P0);
       
        return dataFiltered;
        


    }

};

class MovingAverage{
//Calculate moving average. Return a vector with moving averages
//Give the vector where the moving average is to be takn
//Give the window n for moving average
public:
Data moving_average_brute_force(const Data &mat, int n)
{
    size_t rows= mat.size();
    size_t cols=mat[0].size();
    double sum=0.0,sum1=0.0;
    Data ma(rows,Row(2));
    //Brute Force Implementation of Moving average
    for(size_t i=0; i<rows-n;i++)
    {
        sum=0.0;
       
        //Averages n elements everytime.
        for(size_t j=i;j<i+n;j++)
        {
                sum+=mat[j][0];//Increment sum
               
        }
        //Moving average
        sum=sum/n;
       
    }
    return mat;
}
// Better implementation
double first_average(const Data &mat, int n)
{
    double sum=0.0;
    for(size_t i=0;i<n;i++)
    {
        sum+=mat[i][0];
        cout<< mat[i][0]<<" ";
    }
  
    return sum/n;
}
Data moving_average_improved(const Data &mat, int n)
{
    size_t rows=mat.size();
    //Do the average of just first n elements
    // from second element onwards just add new element and 
    //subtract previous moving average's 1st element.
    double first=mat[0][0];
    double initial=first_average(mat, n);
    double avg,new_term;
    for(size_t i=1; i<rows-n;i++)
    {
        avg=initial*n;
        //Subtract first term
        avg=avg-first;
        first=mat[i][0];
        new_term=mat[i+n-1][0];
        avg+=new_term;
        //Averages n elements everytime.
        initial=avg/n;
        //Moving average
        cout<< "Moving Average:"<< initial<< endl;
    }
    return mat;

} 
};      

//To extract data from file of type CSV
void split_line(string& line, string delim, list<string>& values)
{
    size_t pos = 0;
    while ((pos = line.find(delim, (pos + 1))) != string::npos) {
        string p = line.substr(0, pos);
        values.push_back(p);
        line = line.substr(pos + 1);
    }

    if (!line.empty()) {
        values.push_back(line);
    }
}

int main()
{
    //Read from file
    //The file contains records :
    // Y, X and timestamp . Put it in vector mat.
    ifstream file ( "reports.csv" );
    string value;
    vector<double> B(3);
    list<string> values;
    while ( file.good() )
    {
        getline ( file, value, ',' ); 
        if (value.find('\n') != string::npos) {
            split_line(value, "\n", values);
        } else {
            values.push_back(value);
        }
    }
    size_t s= values.size();
    s=s/3 -1;
    Data mat(s,Row(3));
    size_t m=0;
    list<string>::const_iterator it = values.begin();
    for(int i=0;i<3;i++)it++;
    for (it ; it != values.end(); it) 
    {
        for(int i=0;i<3;i++)
        {
        string tmp = *it;
        //cout << "String val: " <<  tmp << endl;
        double d;
        d = strtod(tmp.c_str(), NULL);
        B[i]=d;
        //cout << "Double val "<<i<<" :" << right << showpoint << B[i] << endl;
        it++;
        }
        mat.at(m)=B;
        //for(int x=0;x<3;x++)
           // cout<< mat[m][x]<<" ";
        //cout<<endl;
        m++;
    }

    MovingAverage mavg;


    //Moving average Brute Force
    int moving_avg_step=5;
    Data ma =mavg.moving_average_brute_force(mat, moving_avg_step);
    //Moving average with improved performance
    Data maImproved =mavg.moving_average_improved(mat, moving_avg_step);
    
    //Kalman Filter
  
    Kalman k;
   
    Data dataKF= k.kalman_init(mat);
    ofstream fileout ( "reports_result.txt" );

    fileout<<"Y,X, timestamp, Y_MA, X_MA, Y_MA_Improved, X_MA_Improved, Y_KalmanFilter,X_KalmanFilter\n";
    //Output with kalman filter result hence total rows+1
    for(size_t i=0;i<s+1;i++)
    {
        if(i<moving_avg_step)
        {
            fileout<<mat[i][0]<<","<<mat[i][1]<<","<<mat[i][2]<<","<<"NULL"<<","<<"NULL"<<","<<"NULL"<<","<<"NULL"<<","<<dataKF[i][0]<<","<<dataKF[i][2]<<endl;

        }
        else if(i==s)
                fileout<<"NULL"<<","<<"NULL"<<","<<mat[i-1][2]+0.5<<","<<"NULL"<<","<<"NULL"<<","<<"NULL"<<","<<"NULL"<<","<<dataKF[i][0]<<","<<dataKF[i][2]<<endl;
            else
                fileout<<mat[i][0]<<","<<mat[i][1]<<","<<mat[i][2]<<","<<ma[i-moving_avg_step][0]<<","<<ma[i-moving_avg_step][1]<<","<<maImproved[i-moving_avg_step][0]<<","<<maImproved[i-moving_avg_step][1]<<","<<dataKF[i][0]<<","<<dataKF[i][2]<<endl;


    }
    fileout.close();
}
