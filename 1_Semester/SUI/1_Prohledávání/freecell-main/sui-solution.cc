// xkucht11, xbiely00

#include "search-strategies.h"
#include <queue>
#include <map>
#include <stack>
#include <iostream>
#include <utility>
#include <set>
#include "memusage.h"
#include <algorithm>
#include <limits.h>

bool operator==(const SearchState &a, const SearchState &b) {
    return a.state_ == b.state_;
}

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state) {
    if (init_state.isFinal()) {
        return {};
    }
    std::queue<std::pair<SearchState, std::vector<SearchAction>>> state_queue;
    std::set<SearchState> visited_states;
    unsigned int reserve_var = 10*mem_limit_/(sizeof(std::pair<SearchState, std::vector<SearchAction>>)+sizeof(SearchState));
    unsigned int i = 0;
    state_queue.push({init_state, {}});
    visited_states.insert(init_state);

    while (!state_queue.empty()) {
        auto [state, actions_so_far] = state_queue.front();
        state_queue.pop();
        auto actions = state.actions();

        if (i % reserve_var) {
            i = 0;
            if (actions.size() != 0) {  // delenie 0
                unsigned int new_res_size = 10*mem_limit_/((sizeof(std::pair<SearchState, std::vector<SearchAction>>)+sizeof(SearchState)+sizeof(actions_so_far)*actions_so_far.size())*actions.size());
                if (reserve_var > new_res_size) {
                    reserve_var = new_res_size;
                }
            }
            if (getCurrentRSS() > 0.99 * mem_limit_) {
                return {};
            }
        }
        for (auto &action : actions) {
            SearchState child_state = action.execute(state);
            std::vector<SearchAction> child_actions = actions_so_far;
            child_actions.emplace_back(action);
            if (child_state.isFinal()) {
                return child_actions;
            }
            if (visited_states.find(child_state) == visited_states.end()) {
                visited_states.insert(child_state);
                state_queue.emplace(std::move(child_state), child_actions);
            }
        }
        i++;
    }
    return {};
}

// pri dfs --dls-limit 4 --easy-mode 33 100 42   o 38% menej stavov, ale 12% dlhsi vypocet. (nejaky hash a hladanie, namiesto kopirovania by prospelo)
int findDepthInStack(std::stack<std::tuple<SearchState, int, bool, std::vector<SearchAction>>> stack, const SearchState& targetState) {
    int min_depth = INT_MAX;
    while (!stack.empty()) {
        SearchState currentState = std::get<0>(stack.top());
        if (currentState == targetState) {
            int currentDepth = std::get<1>(stack.top());
            if (currentDepth < min_depth) {
                min_depth = currentDepth;
            }
        }
        stack.pop(); // Pop the current state and move to the next
    }
    return min_depth;
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state) {
    if (init_state.isFinal() || depth_limit_ == 0) {
        return {};
    }
    std::stack<std::tuple<SearchState, int, bool, std::vector<SearchAction>>> state_stack;
    unsigned int reserve_var = 10*mem_limit_/(sizeof(std::tuple<SearchState, int, bool, std::vector<SearchAction>>)+sizeof(SearchState));
    unsigned int i = 0;
    state_stack.push(std::make_tuple(init_state, 0, false, std::vector<SearchAction>()));

    while (!state_stack.empty()) {
        auto [state, depth, allChildrenProcessed, actions_so_far] = state_stack.top();
        state_stack.pop();
        auto actions = state.actions();

        if (i % reserve_var) {
            if (getCurrentRSS() > 0.99 * mem_limit_) {
                return {};
            } else if  (actions.size() != 0) {  // delenie 0
                unsigned int new_res_size = 10*mem_limit_/((sizeof(std::tuple<SearchState, int, bool, std::vector<SearchAction>>)+sizeof(SearchState)+sizeof(actions_so_far)*actions_so_far.size())*actions.size());
                if (reserve_var > new_res_size) {
                    reserve_var = new_res_size;
                }
            }
            i = 0;
        }
        int child_depth = depth + 1;
        for (auto &action : actions) {
            SearchState child_state = action.execute(state);
            std::vector<SearchAction> child_actions = actions_so_far;
            child_actions.emplace_back(action);
            if (child_state.isFinal()) {
                return child_actions;
            }
            if (child_depth < depth_limit_ && (depth_limit_ > 100 || child_depth < findDepthInStack(state_stack, child_state))) {
                state_stack.emplace(std::make_tuple(child_state, child_depth, false, child_actions));
            }
        }
        i++;
    } 
    return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
    GameState state_copy(state);  // mozno &GameState?
    double card_sequence_score = 0.0;
    for (auto &stack : state_copy.stacks) {
        int stack_size = stack.nbCards();
        int loop_val = 1;
        if (stack_size < 13) {
            loop_val = 14-stack_size;
        }
        while (loop_val <= stack_size) {
            auto card = stack.getCard();  // pop z vrchu
            card_sequence_score += std::abs(loop_val - card->value);
            loop_val++;
        }
    }
    for (const auto &free_cell : state_copy.free_cells) {  // without this, algorithm would probably put everything to freecells, to minimize heuristic.
        auto fc_top = free_cell.topCard();
        if (fc_top.has_value()) {
            card_sequence_score += fc_top->value;
        }  
    }
    return card_sequence_score;
}

using node = std::tuple<double, double, SearchState, std::vector<SearchAction>>;  // f, h, state, actions
struct myComp {
    constexpr bool operator()(const node& a, const node& b) const noexcept {
        return std::get<0>(a) < std::get<1>(b);  // Compare based on the first element
    }
};

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {
    if (init_state.isFinal()) {
        return {};
    }
    std::set<SearchState> visited_states;
    std::priority_queue<node, std::deque<node>, myComp> open;
    unsigned int reserve_var = 100*mem_limit_/(sizeof(node)+sizeof(SearchState));
    unsigned int i = 0;
    double init_h = compute_heuristic(init_state, *heuristic_);
    double min_h = init_h;
    double travel_weight = 3;
    visited_states.insert(init_state);
    open.push(std::make_tuple(init_h, init_h, init_state, std::vector<SearchAction>()));  // f = h,  g = 0;
    int coef = 2;

    while (!open.empty()) {
        auto [f, h, current_state, actions_so_far] = open.top();
        open.pop();
        auto actions = current_state.actions();
        double child_g = f - h + travel_weight;

        if (i % reserve_var) {
            i = 0;
            if (actions.size() != 0) {  // delenie 0
                unsigned int new_res_size = 100*mem_limit_/((sizeof(node)+sizeof(SearchState)+sizeof(actions_so_far)*actions_so_far.size())*actions.size());
                if (reserve_var > new_res_size) {
                    reserve_var = new_res_size;
                }
            }
            if (getCurrentRSS() > 0.99 * mem_limit_) {
                return {};
            }
        }
        
        coef = (h > 100) ? 1.3 : 2.5;

        for (auto &action : actions) {
            SearchState child_state = action.execute(current_state);
            std::vector<SearchAction> child_actions = actions_so_far;
            child_actions.emplace_back(action);
            if (child_state.isFinal()) {
                return child_actions;
            }
            double child_h = compute_heuristic(child_state, *heuristic_);
            if (child_h < h*coef && child_h < 5 * min_h && visited_states.find(child_state) == visited_states.end()) {
                double child_f = child_g + child_h;
                open.push(std::make_tuple(child_f, child_h, child_state, child_actions));
                visited_states.insert(child_state);
                if (child_h < min_h) {
                    min_h = child_h;
                }
            }
        }
        i++;
    }
    return {};
}
