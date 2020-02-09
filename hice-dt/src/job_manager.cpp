/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "job_manager.h"
#include <sstream>

using namespace horn_verification;

bool job_manager::is_leaf(
    const slice &sl,
    bool &label,
    std::unordered_set<datapoint<bool> *> &positive_ptrs,
    std::unordered_set<datapoint<bool> *> &negative_ptrs) {
    assert(sl._left_index <= sl._right_index && sl._right_index < _datapoint_ptrs.size());
    assert(positive_ptrs.empty());
    assert(negative_ptrs.empty());

    // Check which classifications occur
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
            }
            index_of_last_unlabeled = i;
        }

        // Found positively and negatively classified data points, thus no leaf node
        if (found_true && found_false) {
            return false;
        }
    }

    // Only positively or only negatively classified data points (i.e., no unlabeled)
    if (!found_unlabeled) {
        // Since either positive or negative data points occur, found_true indicates which ones occur
        label = found_true; 
        return true;
    }

    // Unlabaled and positive xor negative data points
    if (found_true || found_false) {
        // If we have positive points, we want to label the unlabeled points positive, otherwise negative
        auto &my_pos_ptrs = found_true ? positive_ptrs : negative_ptrs;
        auto &my_neg_ptrs = found_true ? negative_ptrs : positive_ptrs;

        // Try to label all unclassified points
        for (std::size_t i = index_of_first_unlabeled; i <= index_of_last_unlabeled; ++i) {
            if (!_datapoint_ptrs[i]->_is_classified) {
                my_pos_ptrs.insert(_datapoint_ptrs[i]);
            }
        }

        // Query the horn solver to check if this labeling is positive
        bool res = _horn_solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);
        if (res) label = found_true;
        return res;
    }

    // Only unclassified data points

    // Try to turn all data points positive

    for (bool tryLabel = true;; tryLabel = false) {
        auto &fullPoints = tryLabel ? positive_ptrs : negative_ptrs;
        for (std::size_t i = sl._left_index; i <= sl._right_index; ++i) {
            fullPoints.insert(_datapoint_ptrs[i]);
        }

        auto ok = _horn_solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);

        // If labeling satisfies Horn constraints, report leaf with classification true
        if (ok) {
            label = tryLabel;
            return true;
        }

        if (tryLabel) {
            // Next time try to label all points negative
            positive_ptrs.clear();
        } else {
            break;
        }
    }

    // Split is necessary
    return false;
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