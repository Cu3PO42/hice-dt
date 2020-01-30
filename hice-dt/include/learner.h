/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

// C++ includes
#include <vector>

// C includes
#include <cassert>

// Project includes
#include "attributes_metadata.h"
#include "datapoint.h"
#include "decision_tree.h"
#include "horn_constraint.h"
#include "slice.h"

#include "output_visitor.h" // Debug

namespace horn_verification {

/**
 * This class implements the learning algorithm for decision trees.
 *
 * @author Daniel Neider
 *
 * @version 1.0
 */
class learner {
    /// The job manager
    std::unique_ptr<job_manager> _manager;

public:
    /**
     * Creates a new decision tree learner.
     *
     * @param manager The job manager to use
     */
    explicit learner(std::unique_ptr<job_manager> &manager) {
        _manager = std::move(manager);
    }

    /**
     * Learns a decision tree that is consistent with a given set of data points and horn constraints.
     * During the learning process, the data points will be rearranged and altered (the latter happens)
     * in response to resolving unlabaled data points with respect to the horn constraints.
     *
     * @param metadata Meta data describing the datapoints (attributes, number of categories, etc.)
     * @param datapoint_ptrs A vector of pointers to the data points
     * @param horn_constraints
     *
     * @return the learned decision tree
     */
    decision_tree learn(
        const attributes_metadata &metadata,
        std::vector<datapoint<bool> *> &datapoint_ptrs,
        const std::vector<horn_constraint<bool>> &horn_constraints) {
        //
        // If no data points were given, return trivial decision tree (any tree is consistent)
        //
        if (datapoint_ptrs.empty()) {
            auto tree = decision_tree();
            tree._root = new leaf_node(true); // Leaf can have any label

            return tree;
        }

        //
        // Create empty decision tree
        //
        decision_tree tree;

        //
        // Create task list and add initial slice
        //
        _manager->add_slice(slice(0, datapoint_ptrs.size() - 1, &tree._root));

        //
        // Learning loop
        //
        while (_manager->has_jobs()) {
            // Get next job
            const auto next_job = _manager->next_job();

            // Execute job
            const auto new_slices = next_job->run(datapoint_ptrs, metadata);

            // Add new slices
            for (const auto &sl : new_slices) {
                _manager->add_slice(sl);
            }
        }

        assert(tree.root());

        return tree;
    }

    /**
     * Checks whether a decision tree is consistent with a Horn sample given
     * as a set of data points and a set of Horn constraints.
     *
     * @param tree The tree to check
     * @param datapoint_ptrs A set of pointers to the data points
     * @param horn_constraints A set of Horn constraints
     *
     * @returns whether the decision tree is consistent with the Horn sample
     */
    static bool is_consistent(
        decision_tree &tree,
        std::vector<datapoint<bool> *> &datapoint_ptrs,
        const std::vector<horn_constraint<bool>> &horn_constraints) {
        //
        // Empty tree is not consistent with any sample
        //
        if (tree.root() == nullptr) {
            return false;
        }

        //
        // Check data points
        //
        output_visitor v;
        for (const auto &dp : datapoint_ptrs) {
            if (dp->_is_classified && v.output(*tree.root(), *dp) != dp->_classification) {
                return false;
            }
        }

        //
        // Check Horn constraints
        //
        for (const auto &hc : horn_constraints) {
            bool lhs = true;
            for (const auto &dp : hc._premises) {
                if (!v.output(*tree.root(), *dp)) {
                    lhs = false;
                    break;
                }
            }

            // premises are satisfied
            if (lhs && (hc._conclusion == nullptr || !v.output(*tree.root(), *hc._conclusion))) {
                return false;
            }
        }

        return true;
    }
};

}; // End namespace horn_verification

