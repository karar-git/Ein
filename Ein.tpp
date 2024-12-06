#ifndef Ein_tpp
#define Ein_tpp
#include <chrono>
#include <thread>
namespace Ein{

// Q_Learning template

template <typename environment>
Q_Learning<environment>::Q_Learning(int num_bins_per_observation, int n_actions , int n_observations , float epsilon , double epochs , float Learning_rate , float discount_rate , int Burn_in , float Epsilon_reduce , int Epsilon_end )
        : num_bins_per_observation(num_bins_per_observation), n_actions(n_actions), n_observations(n_observations), epsilon(epsilon), epochs(epochs), Learning_rate(Learning_rate), discount_rate(discount_rate), Burn_in(Burn_in), Epsilon_reduce(Epsilon_reduce), Epsilon_End(Epsilon_end), bins(arma::mat(n_observations, num_bins_per_observation, arma::fill::zeros)) {}

template <typename environment>
Q_Learning<environment>::Q_Learning(const Q_Learning<environment>& model)
        : num_bins_per_observation(model.num_bins_per_observation), n_actions(model.n_actions), n_observations(model.n_observations), epsilon(model.epsilon), epochs(model.epochs), Learning_rate(model.Learning_rate), discount_rate(model.discount_rate), Burn_in(model.Burn_in), Epsilon_reduce(model.Epsilon_reduce), Epsilon_End(model.Epsilon_End), bins(model.bins) {}

template <typename environment>
void Q_Learning<environment>::create_bins(const arma::mat& min_max_values){
    if (min_max_values.n_cols != 2) {
        throw std::invalid_argument("min_max_values must have exactly 2 columns, skill issue");
    }

    if (min_max_values.n_rows < n_observations) {
        throw std::invalid_argument(
            "min_max_values must have at least n_observations rows, skill issue");
    }

  double min_val;
  double max_val;
  double bin_step;
    for (int i = 0; i < n_observations; i++) {
        min_val = min_max_values(0, i);  
        max_val = min_max_values(1, i);  
        bin_step = (max_val - min_val) / (num_bins_per_observation - 1);

        for (int j = 0; j < num_bins_per_observation; j++) {
            bins(i, j) = min_val + j * bin_step;
        }
    }
}
 
template <typename environment>
arma::rowvec Q_Learning<environment>::discretize_observation(const arma::rowvec& observations) {
    arma::rowvec binned_observations(n_observations);

    for (size_t i = 0; i < n_observations; i++) {
        bool assigned = false;

        for (size_t j = 0; j < num_bins_per_observation - 1; j++) {
            if (observations(i) >= bins(i, j) && observations(i) < bins(i, j + 1)) {
                binned_observations(i) = j;
                assigned = true;
                break;
            }
        }

        if (!assigned) {
            if (observations(i) >= bins(i, num_bins_per_observation - 1)) {
                binned_observations(i) = num_bins_per_observation - 1;
            } else if (observations(i) < bins(i, 0)) {
                binned_observations(i) = 0;
            }
        }
    }

    return binned_observations;
}


template <typename environment>
arma::uword Q_Learning<environment>::flatten_discrete_state(const arma::rowvec& discrete_state) {
    arma::uword flattened_index = 0;
    arma::uword multiplier = 1;
    for (size_t i = 0; i < discrete_state.n_elem; i++) {
        flattened_index += discrete_state(i) * multiplier;
        multiplier *= num_bins_per_observation;
    }
    return flattened_index;
}

template <typename environment>
int Q_Learning<environment>::epsilon_greedy_action_selection(const arma::mat& q_table, arma::rowvec discrete_state) {
    arma::uword state_index = flatten_discrete_state(discrete_state);
    float random_value = arma::randu<float>();
    if (random_value > epsilon) {
        return arma::index_max(q_table.row(state_index));
    }
    return arma::randi<int>(arma::distr_param(0,n_actions - 1));
}

template <typename environment>
double Q_Learning<environment>::compute_next_q_value(double old_q_value, double next_optimal_q_value) {
    if (done) {
        return old_q_value + Learning_rate * (reward - old_q_value);
    }
    return old_q_value + Learning_rate * (reward + discount_rate * next_optimal_q_value - old_q_value);

}

template <typename environment>
void Q_Learning<environment>::reduce_epsilon() {
    if (epochs >= Burn_in && epochs <= Epsilon_End)
        epsilon -= Epsilon_reduce;
    epsilon = std::max(epsilon, epsilon_min);
}

template <typename environment>
arma::mat Q_Learning<environment>::fit(environment& env) {
    create_bins(env.min_max_values);
    arma::mat q_table = arma::zeros<arma::mat>(std::pow(num_bins_per_observation, n_observations), n_actions);
    std::vector<int> state_visit_count(q_table.n_rows, 0);
    int action;
    arma::rowvec next_state;
    arma::rowvec next_state_discretized;
    double old_q_value;
    double next_optimal_q_value;
    double next_q;
    arma::rowvec initial_state;
    arma::rowvec discretized_state;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        initial_state = env.reset();
        discretized_state = discretize_observation(initial_state);
        done = false;

        while (!done) {
            state_visit_count[flatten_discrete_state(discretized_state)]++;

            action = epsilon_greedy_action_selection(q_table, discretized_state);
            std::tie(next_state, reward, done) = env.step(action);
            next_state_discretized = discretize_observation(next_state);
            old_q_value = q_table(flatten_discrete_state(discretized_state), action);
            next_optimal_q_value = arma::max(q_table.row(flatten_discrete_state(next_state_discretized)));
            next_q = compute_next_q_value(old_q_value, next_optimal_q_value);
            q_table(flatten_discrete_state(discretized_state), action) = next_q;


            discretized_state = next_state_discretized;
        }
        reduce_epsilon();
    }

    // Print state visit counts
//    std::cout << "State visit counts:\n";
 //   for (size_t i = 0; i < state_visit_count.size(); ++i) {
  //      std::cout << "State " << i << ": " << state_visit_count[i] << " visits\n";
   // }

    return q_table;
}


template <typename environment>
void Q_Learning<environment>::play(environment& env,bool computer, const arma::mat& q_table, int rounds) {
        env.render();
        if (computer)
    { for(int i=rounds; i>= 1; i--)
       while (!env.is_done()) 
        {env.computer_play(q_table);
            std::this_thread::sleep_for(std::chrono::seconds(1));  }
    }
        else
                env.people_play();
}
}
#endif // Ein_tpp
