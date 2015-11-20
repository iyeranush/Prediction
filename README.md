# Prediction
PROGRAM:
To Estimate the track of the train

METHODS USED:
1.Simple Moving Average
2.Improved Simple Moving Average with Lesser Computation
3.Kalman Filter

STRUCTURE of THE PROGRAM:
1. Two classes for Kalman Filter and Simple Moving Average
2. Class Consists of Several Functions required for calculating the respective estimation
3. Objects created and function is called with the instance passing the vector of data as argument

ASSUMPTIONS:
1. Assumed that Velocity of the Train is constant
2. Converted the Json file to CSV.
3. Interchanged the columns to the format Y, X, Timestamp
4. Sorted data as per timestamp.
5. Have not implemented the sorting algorithm to sort the timestamp. Did it using Excel. 
Assumed that sensor data or reports come in sorted order.
6. Read from input file.
7. Printted output to output file with format 
(Y,X, timestamp, Y_MovingAverage, X_MovingAverage, Y_MovingAverage_Improved, X_MovingAverage_Improved, Y_KalmanFilter,X_KalmanFilter)
8. Calculated Moving Average for window size=5. Can be changed.
9. Added one extra row in output file for timestamp after 0.5.
10. That is the prediction. It can be changed as per need.
11. Printed NULL in the output whereever there is no observation.

ATTACHED:
1. Input file.
2. Sample Output file
3. Source Code
