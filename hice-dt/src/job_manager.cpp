/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "job_manager.h"

using namespace horn_verification;

bool job_manager::is_leaf(
    const slice &sl,
    bool &label,
    std::unordered_set<datapoint<bool> *> &positive_ptrs,
    std::unordered_set<datapoint<bool> *> &negative_ptrs) {
    assert(sl._left_index <= sl._right_index && sl._right_index < _datapoint_ptrs.size());
    assert(positive_ptrs.empty());
    assert(negative_ptrs.empty());

    //
    // Check which classifications occur
    //
    bool found_true = false;
    bool found_false = false;
    bool found_unlabeled = false;
    std::size_t index_of_first_unlabeled = 0;
    std::size_t index_of_last_unlabeled = 0;

    for (std::size_t i = sl._left_index; i <= sl._right_index; ++i) {
        if (_datapoint_ptrs[i]->_is_classified) {
            if (_datapoint_ptrs[i]->_classification) {
                found_true = true;
            } else {
                found_false = true;
            }
        } else {
            if (!found_unlabeled) {
                found_unlabeled = true;
                index_of_first_unlabeled = i;
                index_of_last_unlabeled = i;
            } else {
                index_of_last_unlabeled = i;
            }
        }

        if (found_true && found_false) {
            break;
        }
    }

    //
    // Found positively and negatively classified data points, thus no leaf node
    //
    if (found_true && found_false) {
        return false;
    }

    //
    // Only positively or only negatively classified data points (i.e., no unlabeled)
    //
    else if (!found_unlabeled) {
        label =
            found_true; // Since either positive or negative data points occur, found_true indicates which ones occur
        return true;
    }

    //
    // Unlabaled and positive data points
    //
    else if (found_true) {
        // Collect unlabeled data points
        for (std::size_t i = index_of_first_unlabeled; i <= index_of_last_unlabeled; ++i) {
            if (!_datapoint_ptrs[i]->_is_classified) {
                positive_ptrs.insert(_datapoint_ptrs[i]);
            }
        }

        // Run Horn solver
        // output_state(positive_ptrs, negative_ptrs, _horn_constraints, "\n---------- + and ? (mark +)", std::cout);
        // horn_solver<bool> solver;
        // auto ok = solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);
        auto ok = _horn_solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);
        // output_state(positive_ptrs, negative_ptrs, _horn_constraints, "\n---------- Solver result: " +
        // std::to_string(ok), std::cout);

        // Labeling satisfies Horn constraints
        if (ok) {
            label = true;
            return true;
        }

        // Labeling does not satisfy Horn constraints
        else {
            // positive_ptrs.clear();
            return false;
        }

    }

    //
    // Unlabaled and negative data points
    //
    else if (found_false) {
        // Collect unlabeled data points
        for (std::size_t i = index_of_first_unlabeled; i <= index_of_last_unlabeled; ++i) {
            if (!_datapoint_ptrs[i]->_is_classified) {
                negative_ptrs.insert(_datapoint_ptrs[i]);
            }
        }

        // Run Horn solver
        // output_state(positive_ptrs, negative_ptrs, _horn_constraints, "\n---------- - and ? (mark -)", std::cout);
        // horn_solver<bool> solver;
        auto ok = _horn_solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);
        // output_state(positive_ptrs, negative_ptrs, _horn_constraints, "\n---------- Solver result: " +
        // std::to_string(ok), std::cout);

        // Labeling satisfies Horn constraints
        if (ok) {
            label = false;
            return true;
        }

        // Labeling does not satisfy Horn constraints
        else {
            // negative_ptrs.clear();
            return false;
        }

    }

    //
    // Only unclassified data points
    //
    else {
        //
        // Try to turn all data points positive
        //
        for (std::size_t i = sl._left_index; i <= sl._right_index; ++i) {
            positive_ptrs.insert(_datapoint_ptrs[i]);
        }

        // Run Horn solver
        // output_state(positive_ptrs, negative_ptrs, _horn_constraints, "\n---------- All ? (mark +)", std::cout);
        // horn_solver<bool> solver;
        auto ok = _horn_solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);
        // output_state(positive_ptrs, negative_ptrs, _horn_constraints, "\n---------- Solver result: " +
        // std::to_string(ok), std::cout);

        // If labeling satisfies Horn constraints, report leaf with classification true
        if (ok) {
            label = true;
            return true;
        }

        //
        // Try to turn all data points negative
        //
        positive_ptrs.clear();
        for (std::size_t i = sl._left_index; i <= sl._right_index; ++i) {
            negative_ptrs.insert(_datapoint_ptrs[i]);
        }

        // Run Horn solver
        // output_state(positive_ptrs, negative_ptrs, _horn_constraints, "\n---------- All ? (mark -)", std::cout);
        // horn_solver<bool> solver1;
        ok = _horn_solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);
        // output_state(positive_ptrs, negative_ptrs, _horn_constraints, "\n---------- Solver result: " +
        // std::to_string(ok), std::cout);

        // If labeling satisfies Horn constraints, report leaf with classification false
        if (ok) {
            label = false;
            return true;
        }

        //
        // Split is necessary
        //
        return false;
    }
}

bool job_manager::unclassified_points_present(
        const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    for (std::size_t i = left_index; i <= right_index; ++i) {
        if (!datapoint_ptrs[i]->_is_classified) {
            return true;
        }
    }
    return false;
}

bool job_manager::positive_points_present(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    for (std::size_t i = left_index; i <= right_index; ++i) {
        if (datapoint_ptrs[i]->_is_classified && datapoint_ptrs[i]->_classification) {
            return true;
        }
    }
    return false;
}

int job_manager::num_points_with_classification(
    const std::vector<datapoint<bool> *> &datapoint_ptrs,
    std::size_t left_index,
    std::size_t right_index,
    bool classification) {
    int count = 0;
    for (std::size_t i = left_index; i <= right_index; ++i) {
        if (datapoint_ptrs[i]->_is_classified && datapoint_ptrs[i]->_classification == classification) {
            count += 1;
        }
    }
    return count;
}