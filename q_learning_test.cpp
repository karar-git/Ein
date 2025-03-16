#include "Ein.hpp"
#include <iostream>
#include <armadillo>
#include <tuple>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <limits>
#include <vector>
#include <algorithm>
class SimpleGame {
public:
    arma::rowvec current_state;
    // The Q_Learning class expects min_max_values as a matrix with exactly 2 columns.
    // Here we define a 2x2 matrix: first column for minimum values, second column for maximum values.
    arma::mat min_max_values;
    int num_bins; // Should match Q_Learning's num_bins_per_observation (here, 10).
    
    SimpleGame() : num_bins(10) {
        current_state = arma::rowvec({1.0, 0.0});
        // Set min_max_values so that:
        // Row 0 (min values): for x and y = 0.0; Row 1 (max values): for x and y = 1.0.
        min_max_values = arma::mat(2, 2);
        min_max_values(0, 0) = 0.0; // min for x
        min_max_values(1, 0) = 1.0; // max for x
        min_max_values(0, 1) = 0.0; // min for y
        min_max_values(1, 1) = 1.0; // max for y
    }

    // Reset the game to the initial state.
    arma::rowvec reset() {
        current_state = arma::rowvec({1.0, 0.0});
        return current_state;
    }

    // The step function takes an action and returns (next_state, reward, done).
    std::tuple<arma::rowvec, double, bool> step(int action) {
        double step_size = 0.1;
        // Action 0: move right (increase x), Action 1: move up (increase y)
        if (action == 0)
            current_state(0) += step_size;
        else if (action == 1)
            current_state(1) += step_size;
        // Clamp state to [0, 1]
        current_state(0) = std::min(current_state(0), 1.0);
        current_state(1) = std::min(current_state(1), 1.0);
        bool done = (current_state(0) >= 1.0 && current_state(1) >= 1.0);
        double reward = done ? 1.0 : -0.1;
        return std::make_tuple(current_state, reward, done);
    }

    // Render the current state.
    void render() {
        std::cout << "Current state: (" << current_state(0) << ", " << current_state(1) << ")\n";
    }

    // Check if the game is done.
    bool is_done() {
        return (current_state(0) >= 1.0 && current_state(1) >= 1.0);
    }

    // Helper function to discretize the current state.
    arma::uword get_discrete_state_index() {
        double bin_width = 1.0 / (num_bins - 1);
        int ix = current_state(0) >= 1.0 ? num_bins - 1 : static_cast<int>(current_state(0) / bin_width);
        int iy = current_state(1) >= 1.0 ? num_bins - 1 : static_cast<int>(current_state(1) / bin_width);
        return ix + iy * num_bins;
    }

    // Use the trained Q-table to pick the best action.
    void computer_play(const arma::mat& q_table) {
        arma::uword state_index = get_discrete_state_index();
        int action = arma::index_max(q_table.row(state_index));
        std::cout << "Computer chooses action: " << action << "\n";
        auto [next_state, reward, done] = step(action);
        render();
    }

    // For human play, prompt the user for an action.
    void people_play() {
        int action;
        std::cout << "Enter action (0 for right, 1 for up): ";
        std::cin >> action;
        auto [next_state, reward, done] = step(action);
        render();
    }
};

// -----------------------------
// Main function to test the game
// -----------------------------
int main() {
    // Seed Armadillo's RNG
    arma::arma_rng::set_seed_random();

    // Create the game environment.
    SimpleGame game;

    // Instantiate Q_Learning with:
    // num_bins_per_observation = 10, n_actions = 2, n_observations = 2, epsilon = 1.0, epochs = 100, etc.
    Ein::Q_Learning<SimpleGame> qlearning(10, 2, 2, 1.0, 100, 0.9, 0.8, 20, 0.01, 80);

    // Train the agent.
    std::cout << "Training the Q-learning agent...\n";
    arma::mat q_table = qlearning.fit(game);

    // Display the trained Q-table.
    //std::cout << "Trained Q-Table:\n" << q_table << "\n";
    game.reset();
    // Let the agent play the game using the trained Q-table.
    std::cout << "Agent playing the game using the trained Q-table:\n";
    qlearning.play(game, true, q_table, 1);

    return 0;
}
