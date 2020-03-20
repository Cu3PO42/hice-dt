#include "util.h"
#include "split.h"
#include "complex_job_manager.h"
#include "simple_job_manager.impl.h"

using namespace horn_verification;

void complex_job_manager::initialize_datapoint_ptrs_to_frac() {
    for (auto & _datapoint_ptr : _datapoint_ptrs) {
        if (_datapoint_ptr->_is_classified) {
            _datapoint_ptrs_to_frac[_datapoint_ptr] = _datapoint_ptr->_classification ? 1.0 : 0.0;
        }
    }
}

void complex_job_manager::update_datapoint_ptrs_to_frac_with_complete_horn_assignments() {
    std::map<datapoint<bool> *, double> _sum_of_datapoint_ptrs_to_frac;

    // Create a list of _datapoint_ptrs which are unclassified
    std::unordered_set<datapoint<bool> *> _unclassified_datapoint_ptrs_stable;
    for (auto & _datapoint_ptr : _datapoint_ptrs) {
        if (!_datapoint_ptr->_is_classified) {
            _unclassified_datapoint_ptrs_stable.emplace(_datapoint_ptr);
        }
        _sum_of_datapoint_ptrs_to_frac[_datapoint_ptr] = 0.0;
    }

    // Now generate complete horn assignments by randomly picking a datapoint from _unclassified_datapoint_ptrs and
    // assigning it True/False followed by label propagation.
    constexpr int numberOfCompleteHornAssignments = 3;
    for (int i = 0; i < numberOfCompleteHornAssignments; i++) {
        auto _unclassified_datapoint_ptrs_temp(_unclassified_datapoint_ptrs_stable);
        std::unordered_set<datapoint<bool> *> positive_ptrs;
        std::unordered_set<datapoint<bool> *> negative_ptrs;

        while (!_unclassified_datapoint_ptrs_temp.empty()) {
            unsigned int itemToAssignClassification = rand() % _unclassified_datapoint_ptrs_temp.size();
            auto item = *std::next(std::begin(_unclassified_datapoint_ptrs_temp), itemToAssignClassification);
            (rand() % 2 == 0 ? positive_ptrs : negative_ptrs).insert(item);

            auto ok = _horn_solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);
            assert(ok);

            // Remove the items present in positive_ptrs and negative_ptrs from _unclassified_datapoint_ptrs_temp
            for (auto it = _unclassified_datapoint_ptrs_temp.begin(); it != _unclassified_datapoint_ptrs_temp.end();) {
                if (positive_ptrs.find(*it) != positive_ptrs.end() || negative_ptrs.find(*it) != negative_ptrs.end()) {
                    it = _unclassified_datapoint_ptrs_temp.erase(it);
                } else {
                    ++it;
                }
            }
        }

        for (auto positive_ptr : positive_ptrs) {
            _sum_of_datapoint_ptrs_to_frac[positive_ptr] += 1.0;
        }
    }
    // Divide _sum_of_datapoint_ptrs_to_frac by 5 and store it to _datapoint_ptrs_to_frac
    for (auto &value : _sum_of_datapoint_ptrs_to_frac) {
        value.second /= numberOfCompleteHornAssignments;
    }

    // Update _datapoint_ptrs_to_frac to _sum_of_datapoint_ptrs_to_frac
    _datapoint_ptrs_to_frac = _sum_of_datapoint_ptrs_to_frac;
}

std::unique_ptr<abstract_job> complex_job_manager::next_job() {
    if (_is_first_split) {
        srand(time(NULL));
        auto sl = _slices.front();
        _slices.pop_front();

        // Check if data points have exactly one categorical attribute
        if (_datapoint_ptrs[sl._left_index]->_categorical_data.size() != 1) {
            throw std::runtime_error("Learner expects exactly one categorical attribute");
        }
        _is_first_split = false;

        return std::unique_ptr<abstract_job>{std::make_unique<categorical_split_job>(sl, 0)};
    } 

    size_t slice_index;
    bool is_weighted = false;
    switch (_node_selection_criterion) {
        case NodeSelection::BFS:
            slice_index = 0; break;

        case NodeSelection::RANDOM:
            slice_index = rand() % _slices.size(); break;

        case NodeSelection::DFS:
            slice_index = _slices.size() - 1; break;

        case NodeSelection::MAX_WEIGHTED_ENTROPY:
        case NodeSelection::MIN_WEIGHTED_ENTROPY:
            is_weighted = true;
        case NodeSelection::MAX_ENTROPY:
        case NodeSelection::MIN_ENTROPY: {
            float best_entropy;
            bool (*cmp)(double, double);
            if (_node_selection_criterion == NodeSelection::MIN_ENTROPY ||
                _node_selection_criterion == NodeSelection::MIN_WEIGHTED_ENTROPY) {
                best_entropy = _node_selection_criterion == NodeSelection::MIN_ENTROPY ? 1.0 : 100000.0;
                cmp = &less<double>;
            } else {
                best_entropy = 0;
                cmp = &greater<double>;
            }
            size_t cur_index = 0;
            for (auto &_slice : _slices) {
                auto entropy_val = is_weighted
                                       ? weighted_entropy(_datapoint_ptrs, { _slice._left_index, _slice._right_index })
                                       : entropy(_datapoint_ptrs, { _slice._left_index, _slice._right_index });
                if (cmp(entropy_val, best_entropy)) {
                    best_entropy = entropy_val;
                    slice_index = cur_index;
                }
                ++cur_index;
            }

            break;
        }
        default:
            assert(false);
    }
    assert(slice_index >= 0 && slice_index < _slices.size());

    auto it = _slices.begin();
    advance(it, slice_index);
    auto sl = *it;
    _slices.erase(it);

    if (_entropy_computation_criterion == EntropyComputation::HORN_ASSIGNMENTS) {
        // Clear _datapoint_ptrs_to_frac from a previous iteration
        _datapoint_ptrs_to_frac.clear();

        // Initialize the _datapoint_ptrs_to_frac map using the classified points in _datapoint_ptrs
        initialize_datapoint_ptrs_to_frac();

        if (unclassified_points_present(_datapoint_ptrs, { sl._left_index, sl._right_index })) {
            // Update _datapoint_ptrs_to_frac with randomly selected complete horn assignments
            update_datapoint_ptrs_to_frac_with_complete_horn_assignments();
        }
    }

    bool label = false; // label is unimportant (if is_leaf() returns false)
    std::unordered_set<datapoint<bool> *> positive_ptrs, negative_ptrs;

    auto can_be_turned_into_leaf = is_leaf(sl, label, positive_ptrs, negative_ptrs);

    // Slice can be turned into a leaf node
    if (can_be_turned_into_leaf) {
        return std::make_unique<leaf_creation_job>(
            sl, label, std::move(positive_ptrs), std::move(negative_ptrs));
    }
    // Slice needs to be split
    else {
        return find_best_split(sl);
    }
}

double complex_job_manager::entropy(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, const index_list &indices) {
    if (_entropy_computation_criterion == EntropyComputation::HORN_ASSIGNMENTS) {
        double count_f = 0.0;
        double count_t = 0.0;
        double sum = 0;

        for (auto &pair : indices) {
            for (std::size_t i = pair.left; i <= pair.right; ++i) {
                // assert that datapoint_ptrs[i] is present in the map _datapoint_ptrs_to_frac
                auto it = _datapoint_ptrs_to_frac.find(datapoint_ptrs[i]);
                assert(it != _datapoint_ptrs_to_frac.end());
                count_t += it->second;
            }
            sum += pair.right - pair.left + 1;
        }

        count_f = sum - count_t;

        // TODO: the following is the same as in the default case
        // std::cout << "sum=" << sum << std::endl;
        double p_t = ((double)(count_t) / sum);
        // std::cout << "p_t=" << p_t << std::endl;
        double p_f = ((double)(count_f) / sum);
        // std::cout << "p_f=" << p_f << std::endl;

        double entropy_t = count_t == 0.0 ? 0 : p_t * log2(p_t);
        double entropy_f = count_f == 0.0 ? 0 : p_f * log2(p_f);

        return -(entropy_t + entropy_f);

    } else if (_entropy_computation_criterion == EntropyComputation::DEFAULT_ENTROPY ||
        _entropy_computation_criterion == EntropyComputation::PENALTY) {
        return simple_job_manager::entropy(datapoint_ptrs, indices);
    }
    // _entropy_computation_criterion should be one of the two implemented criterions
    // Control should never reach here!
    assert(false);
    return 0.0;
}

unsigned int complex_job_manager::num_classified_points(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, const index_list &indices) {
    if (_entropy_computation_criterion == EntropyComputation::DEFAULT_ENTROPY ||
        _entropy_computation_criterion == EntropyComputation::PENALTY) {
        return simple_job_manager::num_classified_points(datapoint_ptrs, indices);
    } else if (_entropy_computation_criterion == EntropyComputation::HORN_ASSIGNMENTS) {
        size_t sum = 0;
        for (auto &pair : indices) {
            sum += pair.right - pair.left + 1;
        }
        return sum;
    }
    assert(false);
    return 0;
}

std::unique_ptr<abstract_job> complex_job_manager::find_best_split(const slice &sl) {
    return find_best_split_base<cat_split, complex_int_split, complex_job_manager>(sl);
}

std::pair<int, int> complex_job_manager::penalty(
    std::size_t left_index,
    std::size_t cur_index,
    std::size_t right_index) {
    int _left2right = 0;
    int _right2left = 0;
    for (const auto &horn_clause : _horn_constraints) {
        enum class Position { out_of_scope, left, right };
        Position conclusion = Position::out_of_scope;
        int num_premise_left = 0;
        int num_premise_right = 0;

        // OPTIMIZE: this is an O(nm) loop, but it really should be O(n+m) or at least O(n log n+m log m)
        // for i ranging from left_index to cur, loop over premises and conclusion
        auto left_overlap = count_overlap(horn_clause, left_index, cur_index);
        auto right_overlap = count_overlap(horn_clause, cur_index + 1, right_index);
        if (left_overlap.second) {
            _right2left += right_overlap.first;
        } else if (right_overlap.second) {
            _left2right += left_overlap.first;
        }
    }
    return { _left2right, _right2left };
}

std::pair<size_t, bool> complex_job_manager::count_overlap(
    horn_constraint<bool> horn_clause,
    std::size_t start_index,
    std::size_t end_index) {
    size_t num_premises = 0;
    bool has_conclusion = false;

    for (std::size_t i = start_index; i <= end_index; ++i) {
        for (const auto dp : horn_clause._premises) {
            if (dp == _datapoint_ptrs[i] && !_datapoint_ptrs[i]->_is_classified) {
                num_premises++;
            }
        }
        if (_datapoint_ptrs[i] == horn_clause._conclusion && !_datapoint_ptrs[i]->_is_classified) {
            has_conclusion = true;
        }
    }

    return { num_premises, has_conclusion };
}