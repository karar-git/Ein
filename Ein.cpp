#include "Ein.hpp"
#include <armadillo>
#include <iostream>
#include <omp.h>
namespace Ein{
//preprocessing 
arma::mat Data::read_csv(const std::string& file_path, bool copy){
  arma::mat data;

  if (!data.load(file_path, arma::csv_ascii)) {
    throw std::runtime_error("Failed to read the CSV file: " + file_path);
  }

  if (copy) {
    arma::mat data_copy = data; 
    return data_copy;
  }

  return data; 
}

void Data::head(const arma::mat& data, size_t n){
  if (n < data.n_rows) {
        std::cout << data.rows(0, n-1) << std::endl;
    } else {
        std::cout << "The matrix has fewer than " << n << " rows." << std::endl;
    }
}

void Data::shuffle(arma::mat& data, unsigned int random_state ){
    arma::arma_rng::set_seed(random_state);
    arma::uvec indices = arma::randperm(data.n_rows);
    data=data.rows(indices);
}
std::tuple<arma::mat, arma::mat> Data::split(const arma::mat& data, int target_col){
    arma::mat target = data.col(target_col);
    arma::mat features = data;
    features.shed_col(target_col);  
    return std::make_tuple(features, target);
}

std::tuple<arma::mat, arma::mat, arma::mat, arma::mat> Data::train_test_split(const arma::mat& features, const arma::mat& targets, float size_train /*arma::mat stratify= arma::mat()*/
){
  size_t train_indices= features.n_rows * 0.8;
  arma::mat x_train= features.rows(0,train_indices - 1);
  arma::mat x_test= features.rows(train_indices, features.n_rows-1);
  arma::mat y_train= targets.rows(0,train_indices-1);
  arma::mat y_test= targets.rows(train_indices, targets.n_rows-1);
  return std::make_tuple(x_train, x_test, y_train, y_test);
}

double Data::sum_squared(const arma::mat& first_values, const arma::mat& second_values) {
    if (first_values.size() != second_values.size()) {
        throw std::invalid_argument("Inputs must have the same dimensions. U have skill issue <3");
    }
    arma::mat diff = first_values - second_values;
    return arma::accu(arma::square(diff));  
}

double Data::sum_abs(const arma::mat& first_values, const arma::mat& second_values) {
    if (first_values.size() != second_values.size()) {
        throw std::invalid_argument("Inputs must have the same dimensions. U have skill issue <3");
    }
    arma::mat diff = arma::abs(first_values - second_values);
    return arma::accu(diff);  
}



double Data::r_squared(arma::mat targets, arma::mat outputs) {
    double SS_residuals = sum_squared(targets, outputs);
    arma::rowvec mean_targets = arma::mean(targets, 0);
    arma::mat mean_targets_broadcasted = arma::repmat(mean_targets, targets.n_rows, 1);  
    double SS_total = sum_squared(targets, mean_targets_broadcasted);
  return 1.0 - (SS_residuals / SS_total);
}

double Data::MAE(arma::mat targets, arma::mat outputs){
  return sum_abs(targets, outputs) / targets.n_rows;
}

double Data::MSE(arma::mat targets, arma::mat outputs){
  return sum_squared(targets, outputs) / targets.n_rows;
}
//Scalers stuff
//Standard_Scaler

void Standard_Scaler::fit(const arma::mat& features){
  means = arma::mean(features, 0);
  deviations = arma::stddev(features, 0, 0);
}

arma::mat Standard_Scaler::transform(arma::mat& features, bool in_place) {
    arma::mat* target = &features;
    if (!in_place) {
        target = new arma::mat(features);
    }

    for (size_t col = 0; col < features.n_cols; ++col) {
        if (deviations(col) != 0) {
            target->col(col) = (target->col(col) - means(col)) / deviations(col);
        }
    }

    if (!in_place) {
        arma::mat result = *target;
        delete target;
        return result;
    }

    return features;
}

void Standard_Scaler::show_means(){
  std::cout << means << std::endl;
}

void Standard_Scaler::show_deviations(){
  std::cout << deviations << std::endl;
}

//Min_Max_Scaler

void Min_Max_Scaler::fit(const arma::mat& features){
  Mins = arma::min(features, 0);
  Maxs = arma::max(features, 0);
}

arma::mat Min_Max_Scaler::transform(arma::mat& features, bool in_place){
  if(!in_place){
    arma::mat features_2 = features;
    for (size_t col = 0; col < features.n_cols; col++) {
        if (Maxs(col) == Mins(col)) {
            features_2.col(col) = 0;  
        } else {
            features_2.each_col() -= Mins;
            features_2.each_col() /= (Maxs - Mins);  
        }
    }
    return features_2;
  }
  else{
    for (size_t col = 0; col < features.n_cols; col++) {
        if (Maxs(col) == Mins(col)) {
            features.col(col) = 0;  
        } else {
            features.each_col() -= Mins;
            features.each_col() /= (Maxs - Mins);  
        }
    }
    return features; 
  }
}

void Min_Max_Scaler::show_Mins(){
  std::cout << Mins << std::endl;
}

void Min_Max_Scaler::show_Maxs(){
  std::cout << Maxs << std::endl;
}

//Supervised Learning
//Linear Regression model

Linear_Regression::Linear_Regression(float Learning_rate , int epochs )
        : Learning_rate(Learning_rate), epochs(epochs) {}
Linear_Regression::Linear_Regression(const Linear_Regression& model)
        : Learning_rate(model.Learning_rate), epochs(model.epochs), weights(model.weights), intercept(model.intercept) {}

void Linear_Regression::fit(const arma::mat& features, const arma::mat& targets,
                            bool initialization , int batch_size , int print_every_n_epochs ,  double lambda, bool Early_stopping, int patience) {
  if (initialization) {
    intercept = 0.0;
    weights = arma::randn<arma::mat>(features.n_cols, 1) * 0.01;
    }

  int n_samples = features.n_rows;
    int n_batches;
    if (batch_size >= 1 && batch_size <= features.n_rows) n_batches = (n_samples + batch_size - 1) / batch_size;
    else {std::cerr<<"the number of batches isn't valid, but as we so helpful people :)), we will make the batch size = number of observations"<<std::endl;
    n_batches = features.n_rows;
    }
  arma::mat errors;
    double best_mse = std::numeric_limits<double>::infinity();
    int epochs_without_improvement = 0;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    arma::mat weight_updates(weights.n_rows, 1, arma::fill::zeros);
    double intercept_update = 0.0;

    for (int batch = 0; batch < n_batches; ++batch) {
      int start_idx = batch * batch_size;
      int end_idx = std::min(start_idx + batch_size, n_samples);

      arma::mat batch_features = features.rows(start_idx, end_idx - 1);
      arma::mat batch_targets = targets.rows(start_idx, end_idx - 1);

      arma::mat batch_errors = batch_targets - predict(batch_features);
      weight_updates = batch_features.t() * batch_errors / batch_features.n_rows;
        weight_updates += (2 * lambda / batch_features.n_rows) * weights;
      intercept_update = arma::as_scalar(arma::sum(batch_errors) / batch_features.n_rows);

    weights += Learning_rate * weight_updates;
    intercept += Learning_rate * intercept_update;}

    if (!(epoch % print_every_n_epochs)) {
      errors = targets - predict(features);
      std::cout << "Epoch " << epoch << " MSE: "
                << arma::dot(errors, errors) / features.n_rows << std::endl;
    }
    if (Early_stopping) {
        double mse = arma::dot(errors, errors) / features.n_rows;
        if (mse > best_mse) {
            epochs_without_improvement++;
            if (epochs_without_improvement >= patience) {
                std::cout << "Early stopping at epoch " << epoch << std::endl;
                break;
            }
        } else {
            best_mse = mse;
            epochs_without_improvement = 0;
         }
    }
  }
}


arma::mat Linear_Regression::predict(const arma::mat& features){
  return features * weights + intercept;
}



//Unsupervised_Learning
//PCA

PCA::PCA(size_t n_components)
        : n_components(n_components) {}
PCA::PCA(const PCA& model)
        : n_components(model.n_components), components(model.components) {}

void PCA::fit(const arma::mat& scaled_features){
  arma::mat covariance_matrix = arma::cov(scaled_features);
  arma::eig_sym(eigenvalues, eigenvectors, covariance_matrix);
  sorted_indices = arma::sort_index(eigenvalues, "descend");
}

arma::mat PCA::transform(const arma::mat& scaled_features, size_t number){
  n_components = number;
  components = eigenvectors.cols(sorted_indices.head(n_components));
  return scaled_features * components;
}

void PCA::show(){
    std::cout << "just use python :)" << std::endl;
    std::cout << "Eigenvalues (Sorted in Descending Order):\n" << std::endl;
    for (size_t i = 0; i < sorted_indices.n_elem; i++) {
    std::cout << "  Column " << sorted_indices[i] << " -> Eigenvalue: " << eigenvalues[sorted_indices[i]] << std::endl;
    }}
void PCA::show_eigens(){
    std::cout << "Eigenvectors corresponding to the eigenvalues:\n" << std::endl;
    for (size_t i = 0; i < sorted_indices.n_elem; i++) {
        std::cout << "  Eigenvector for Column " << sorted_indices[i] 
                  << " -> " << eigenvectors.col(sorted_indices[i]).t() << std::endl;

  }
  std::cout << "What about giving Vim a try instead of using IDEs? 🙃" << std::endl;
}

}
