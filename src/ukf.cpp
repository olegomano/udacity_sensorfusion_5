#include "ukf.h"
#include "Eigen/Dense"
#include <stdio.h>
#include <iostream>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::VectorXd;


void normalizeAngle(VectorXd& v, int indx){
 // v(indx) = fmod(v(indx),M_PI);
 while (v(indx)> M_PI) v(indx)-=2.*M_PI;
 while (v(indx)<-M_PI) v(indx)+=2.*M_PI;
}


VectorXd genWeights(double lambda,int augSize){
  VectorXd weights = VectorXd(2*augSize+1);
  
  weights(0) = lambda/(lambda+augSize);
  for (int i=1; i<weights.size(); ++i) {  // 2n+1 weights
    double weight = 0.5/(augSize+lambda);
    weights(i) = weight;
  }

  return weights;
}

VectorXd stateToRadar(const VectorXd& state){
  VectorXd radar = VectorXd(3);
  double p_x = state(0);
  double p_y = state(1);
  double v   = state(2);
  double yaw = state(3);

  double v1 = cos(yaw)*v;
  double v2 = sin(yaw)*v;

    // measurement model
  radar(0) = sqrt(p_x*p_x + p_y*p_y);                       // r
  radar(1) = atan2(p_y,p_x);                                // phi
  radar(2) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  return radar;
}

VectorXd stateToLidar(const VectorXd& state){
  VectorXd lidar = VectorXd(2);
  lidar(0) = state(0);
  lidar(1) = state(1);
  return lidar;
}

MatrixXd crossCorellate(const VectorXd& stateMean, const MatrixXd& stateSigma, const VectorXd& measureMean, const MatrixXd& measureSigma, const VectorXd& w){
  MatrixXd cross = MatrixXd(stateMean.rows(),measureMean.rows());
  cross.fill(0);
  for(int i = 0; i < stateSigma.cols(); i++){
    VectorXd diffState = stateSigma.col(i)   - stateMean;
    VectorXd diffMeas  = measureSigma.col(i) - measureMean;
    normalizeAngle(diffState,3);
    if(diffMeas.size() == 3){
      normalizeAngle(diffMeas,1);
    }
    cross += w(i)*diffState*diffMeas.transpose();
  }
  return cross;
}


std::tuple<VectorXd,MatrixXd> sigmaToCov(const MatrixXd& sigma, const VectorXd& w){
  VectorXd mean = VectorXd(sigma.rows());
  MatrixXd cov  = MatrixXd(sigma.rows(),sigma.rows());

  cov.fill(0);
  mean.fill(0);

  for (int i = 0; i < sigma.cols(); ++i) {  // iterate over sigma points
    mean+= (w(i) * sigma.col(i));
  }

  for (int i = 0; i < sigma.cols(); ++i) {  // iterate over sigma points
      VectorXd d = sigma.col(i) - mean;
      if(d.size() == 5){
        normalizeAngle(d,3);
      }else if(d.size() == 3){
        normalizeAngle(d,1);
      }
      cov+= (w(i) * d * d.transpose());
  }

  return std::make_tuple(mean,cov);

}

MatrixXd genSigmaPoints(const VectorXd& mean, const MatrixXd& cov,double lambda, double stdA,double stdYawd){
  VectorXd meanAug = VectorXd(mean.size() + 2);
  MatrixXd covAug  = MatrixXd(cov.rows() + 2, cov.cols() + 2);
  covAug.fill(0);
  meanAug.fill(0);
  //meanAug(5) = stdA;
  //meanAug(6) = stdYawd;
  for(int i = 0; i < mean.size(); i++){
    meanAug(i) = mean(i);
  }
  for(int i = 0; i < cov.rows(); i++){
    for(int y = 0; y < cov.cols(); y++){
        covAug(i,y) = cov(i,y);
    }
  }


  covAug(5,5) = stdA*stdA;
  covAug(6,6) = stdYawd*stdYawd;
  MatrixXd L = covAug.llt().matrixL();
  

  MatrixXd sigmaPoints = MatrixXd(covAug.rows(), 2 * covAug.rows() + 1);
  sigmaPoints.fill(0);
  sigmaPoints.col(0) = meanAug;
  
  for (int i = 0; i< meanAug.size(); ++i) {
    sigmaPoints.col(i+1)                = meanAug + sqrt(lambda+meanAug.size()) * L.col(i);
    sigmaPoints.col(i+1+meanAug.size()) = meanAug - sqrt(lambda+meanAug.size()) * L.col(i);  
  }

  return sigmaPoints;
}

VectorXd predictState(const VectorXd& state, double dt){
  VectorXd pState = VectorXd(5);
  double p_x      = state(0);
  double p_y      = state(1);
  double v        = state(2);
  double yaw      = state(3);
  double yawd     = state(4);
  double nu_a     = state(5);
  double nu_yawdd = state(6);

    // predicted state values
  double px_p, py_p;

  // avoid division by zero
  if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*dt) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*dt) );
  } else {
      px_p = p_x + v*dt*cos(yaw);
      py_p = p_y + v*dt*sin(yaw);
  }

  double v_p = v;
  double yaw_p = yaw + yawd*dt;
  double yawd_p = yawd;
   // add noise
  px_p = px_p + 0.5*nu_a*dt*dt * cos(yaw);
  py_p = py_p + 0.5*nu_a*dt*dt * sin(yaw);
  v_p = v_p + nu_a*dt;

  yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
  yawd_p = yawd_p + nu_yawdd*dt;
   // write predicted sigma point into right column
  pState(0) = px_p;
  pState(1) = py_p;
  pState(2) = v_p;
  pState(3) = yaw_p;
  pState(4) = yawd_p;
  return pState;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.35;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  initRadar       = false;
  initLidar       = false;
  lambda_ = -4;
  x_.fill(0);
  P_(0,0)  = 0.65;
  P_(1,1)  = 1.5;
  P_(2,2)  = 0.76;
  P_(3,3)  = 0.13;
  P_(4,4)  = 0.085;
  radarNoise = MatrixXd(3,3);
  lidarNoise = MatrixXd(2,2);

  radarNoise <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;

  lidarNoise << std_laspx_*std_laspx_ , 0,
                0, std_laspy_*std_laspy_;

  time_us_ = 0; 

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if(!is_initialized_){
    time_us_        = meas_package.timestamp_;
    if(meas_package.sensor_type_ == MeasurementPackage::LASER){
      x_.fill(0);
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      initLidar = true;
    }
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      x_(2)     = meas_package.raw_measurements_(2) * cos(meas_package.raw_measurements_(1)); 
      initRadar = true;
    }
    if(initRadar && initLidar){
      is_initialized_ = true;
    }
    return;
  }


  double dt   = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_    = meas_package.timestamp_;
  VectorXd w  = genWeights(lambda_,x_.size() + 2);
  std::cout << "dt     " << std::endl << dt << std::endl;
  std::cout << "Meas   " << std::endl << meas_package.raw_measurements_ << std::endl;
  std::cout << "Mean   " << std::endl << x_ << std::endl;
  std::cout << "Cov    " << std::endl << P_ << std::endl;
  

  MatrixXd sigma  = genSigmaPoints(x_,P_,lambda_,std_a_,std_yawdd_);
  MatrixXd sigmaP = MatrixXd(5,sigma.cols());
  for(int i = 0; i < sigma.cols(); i++){
    VectorXd p  = predictState(sigma.col(i),dt);
    normalizeAngle(p,3);
    sigmaP.col(i) = p;
  }

  std::tuple<VectorXd,MatrixXd> meanCovPredicted = sigmaToCov(sigmaP,w);
  std::tuple<VectorXd,MatrixXd> meanCovMeasure;
  MatrixXd sigmaM;
  switch(meas_package.sensor_type_){
    case MeasurementPackage::LASER:
      sigmaM = MatrixXd(2,sigmaP.cols());
      sigmaM.fill(0);
      for(int i = 0; i < sigma.cols(); i++){
        sigmaM.col(i)(0) = sigma.col(i)(0);
        sigmaM.col(i)(1) = sigma.col(i)(1);
      }
      meanCovMeasure = sigmaToCov(sigmaM,w);
      std::get<1>(meanCovMeasure)+=lidarNoise;
      break;
    
    case MeasurementPackage::RADAR:
      sigmaM = MatrixXd(3,sigmaP.cols());
      for(int i = 0; i < sigma.cols(); i++){
        sigmaM.col(i) = stateToRadar(sigmaP.col(i));
      }
      meanCovMeasure = sigmaToCov(sigmaM,w);
      std::get<1>(meanCovMeasure)+=radarNoise;
      normalizeAngle(std::get<0>(meanCovMeasure),1);
      break;
  }

  MatrixXd crossCor = crossCorellate(std::get<0>(meanCovPredicted),sigmaP,std::get<0>(meanCovMeasure),sigmaM,w);
  MatrixXd gain     = crossCor * std::get<1>(meanCovMeasure).inverse();

  VectorXd meanUpdate = gain*(meas_package.raw_measurements_ - std::get<0>(meanCovMeasure));
  MatrixXd covUpdate  = gain * std::get<1>(meanCovMeasure) * gain.transpose();

  x_ = std::get<0>(meanCovPredicted) + meanUpdate;
  P_ = std::get<1>(meanCovPredicted) - covUpdate;

  /*
  std::cout << "Sigma   " << std::endl << sigma  << std::endl;
  std::cout << "Sigma P " << std::endl << sigmaP << std::endl;
  std::cout << "Sigma M " << std::endl << sigmaM << std::endl;
  std::cout << "Mean  P " << std::endl << std::get<0>(meanCovPredicted) << std::endl;
  std::cout << "Mean  M " << std::endl << std::get<0>(meanCovMeasure) << std::endl;
  std::cout << "Cov   M " << std::endl << std::get<1>(meanCovMeasure) << std::endl;
  std::cout << "cross   " << std::endl << crossCor << std::endl;
  std::cout << "gain    " << std::endl << gain     << std::endl;
  
  std::cout << meanUpdate << std::endl << std::endl << std::endl;
  std::cout << covUpdate  << std::endl << std::endl << std::endl;
*/

  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  std::cout << " Prediction " << delta_t << std::endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}
