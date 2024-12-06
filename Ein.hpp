#ifndef Ein_hpp
#define Ein_hpp

#include <armadillo>
#include <tuple>
namespace Ein{
class Data {
public:
    static arma::mat read_csv(const std::string& file_path, bool copy = true);
    static void head(const arma::mat& data, size_t n=5);
    static void shuffle(arma::mat& data, unsigned int random_state = 0);
    static std::tuple<arma::mat, arma::mat> split(const arma::mat& data, int target_col);
    static std::tuple<arma::mat, arma::mat, arma::mat, arma::mat> train_test_split(const arma::mat& features, const arma::mat& targets, float size_train = 0.8);
    static double sum_squared(const arma::mat& first_values, const arma::mat& second_values);
    static double sum_abs(const arma::mat& first_values, const arma::mat& second_values);
    static double r_squared(arma::mat targets, arma::mat outputs);
    static double MAE(arma::mat targets, arma::mat outputs);
    static double MSE(arma::mat targets, arma::mat outputs);
};

class Standard_Scaler {
public:
    arma::rowvec means, deviations;
    void fit(const arma::mat& features);
    arma::mat transform(arma::mat& features, bool in_place = false);
    void show_means();
    void show_deviations();
};

class Min_Max_Scaler {
public:
    arma::rowvec Mins, Maxs;
    void fit(const arma::mat& features);
    arma::mat transform(arma::mat& features, bool in_place = false);
    void show_Mins();
    void show_Maxs();
};

class Linear_Regression {
public:
    float Learning_rate;
    int epochs;
    arma::mat weights;
    double intercept;
    Linear_Regression(float Learning_rate = 0.01, int epochs = 40);

    Linear_Regression(const Linear_Regression& model);

    void fit(const arma::mat& features, const arma::mat& targets, //bool opt_meth = adam
             bool initialization = true, int batch_size=64, int print_every_n_epochs=10,  double lambda = 0.0, bool Early_stopping = false, int patience = 0);
    arma::mat predict(const arma::mat& features);
};

class PCA {
public:
    size_t n_components;
    arma::mat components;
    arma::uvec sorted_indices;
    arma::vec eigenvalues;
    arma::mat eigenvectors;

    PCA(size_t n_components = 3);
    PCA(const PCA& model);

    void fit(const arma::mat& scaled_features);
    arma::mat transform(const arma::mat& scaled_features){
    return transform(scaled_features, n_components);

    }
    arma::mat transform(const arma::mat& scaled_features, size_t number);
    void show();
    void show_eigens();
};

template <typename environment>
class Q_Learning {
public:
    int num_bins_per_observation, n_actions, n_observations, Burn_in, Epsilon_End;
    bool done = false;
    float epsilon, Learning_rate, discount_rate, Epsilon_reduce;
    float epsilon_min = 0.05;
    double reward, epochs;
    arma::mat bins;

    Q_Learning(int num_bins_per_observation = 10, int n_actions = 0, int n_observations = 0, float epsilon = 1, double epochs = 100, float Learning_rate = 0.9, float discount_rate = 0.8, int Burn_in = 20, float Epsilon_reduce = 0.01, int Epsilon_end = 80);

    Q_Learning(const Q_Learning& model);
    void create_bins(const arma::mat& min_max_values);
    arma::rowvec discretize_observation(const arma::rowvec& observations);
    arma::uword flatten_discrete_state(const arma::rowvec& discrete_state);
    int epsilon_greedy_action_selection(const arma::mat& q_table, arma::rowvec discrete_state);
    double compute_next_q_value(double old_q_value, double next_optimal_q_value);
    void reduce_epsilon();
    arma::mat fit(environment& env);
    void play(environment& env, bool computer = 1, const arma::mat& q_table= arma::mat(1,1,arma::fill::zeros), int rounds = 1);
};

}
#include "Ein.tpp"
#endif

