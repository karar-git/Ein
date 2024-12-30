#include "Ein.hpp"
#include <armadillo>
#include <utility>
#include <filesystem>

using namespace std;

int main(){
      //detect the file path
    filesystem::path sourceDir = filesystem::path(__FILE__).parent_path();

    arma::mat data=Ein::Data::read_csv(sourceDir / "California_houses.csv");
  cout<<"Date imported!"<<endl;

    Ein::Data::shuffle(data);

    auto [features, targets] = Ein::Data::split(data, data.n_cols - 2);

    data.clear();

    auto [x_train, x_test, y_train, y_test] = Ein::Data::train_test_split(features, targets, 0.75);

    features.clear(); targets.clear();
    Ein::Standard_Scaler sc;
    sc.fit(x_train);
    sc.transform(x_train, /* in_place= ,*/ true);
    Ein::PCA PCA_model;
    PCA_model.fit(x_train);
    PCA_model.show();
    auto x_train_PCA=PCA_model.transform(x_train, x_train.n_cols);
    Ein::Linear_Regression lr(0.01, 250);
    lr.fit(x_train_PCA, y_train, true , 256, 10, 1, true, 3);
    sc.transform(x_test, true);
    x_test = PCA_model.transform(x_test, x_train.n_cols );
    auto predicted_values = lr.predict(x_test);
    cout<<"\n R^2="<<Ein::Data::r_squared(y_test, predicted_values);
    cout<<"\n Mean Absolute Error="<<Ein::Data::MAE(y_test, predicted_values)<<endl;
  cout<<"the predicted values:\n";
    Ein::Data::head(predicted_values); cout<<"\n";
  cout<<"the actual values:\n";
    Ein::Data::head(y_test);
    return 0;
}
