/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

#include "simple_job_manager.h"

namespace horn_verification {

/**
 * Implements a complex job manager by deriving from the simple_job_manager.
 *
 * A job manager implements complex heuristics for
 * <ul>
 *   <li> deciding which node to process next; and</li>
 *   <li> scoring a node with respect to the classification of datapoints in that node (aka entropy computation that
 *        takes into account the horn constraints) </li>
 * </ul>
 *
 * A job manager specializes the following methods:
 * <ul>
 *   <li> next_job(): This method returns the job the manager wants to
 *        process next. Different heurisitcs are invoked depending on the member enum variable
 *        heuristic_for_node_selection. </li>
 *   <li> entropy(): This method returns an entropy score for a given set of datapoints. Different heuristics compute
 *        the entropy differently. Most heuristics take into account the horn constraints while computing this
 *        score.</li>
 * </ul>
 *
 * The learner uses the job manager in the following way:
 * <ol>
 *   <li> The learning algorithm adds slices using add_slice(). </li>
 *   <li> The job manager decides which \ref slice to process next and what
 *        to do with the \ref slice (create a leaf or split).</li>
 *   <li> The learning algorithm calls next_job() to retrieve the next job
 *        and processes it.</li>
 * </ol>
 * This process repeats until all jobs have been processed (i.e., has_jobs()
 * returns \c false).
 */
class complex_job_manager : public job_manager {
private:
    NodeSelection _node_selection_criterion;
    EntropyComputation _entropy_computation_criterion;
    ConjunctiveSetting _conjunctive_setting;

    // A map from the set of (pointers to) data points to a fractional value between 0 (negative) and 1 (positive).
    // Fractional value for a datapoint is the likelihood of the given point to be assigned positive in some randomly
    // chosen completion of the Horn assignment. Only used when _entropy_computation_criterion == HORN_ASSIGNMENTS.
    std::map<datapoint<bool> *, double> _datapoint_ptrs_to_frac;

public:
    /**
     * Creates a new complex job manager.
     *
     * @param datapoint_ptrs A reference to the set of (pointers to) data points over which to work
     * @param horn_constraints A reference to the horn constraints over which to work
     * @param solver A reference to the Horn solver to use
     * @param node_selection_criterion Node selection heuristic to be used while building the tree
     * @param entropy_computation_criterion Criterion for scoring a node/slice a la entropy
     */
    complex_job_manager(
        std::vector<datapoint<bool> *> &datapoint_ptrs,
        const std::vector<horn_constraint<bool>> &horn_constraints,
        horn_solver<bool> &solver,
        NodeSelection node_selection_criterion,
        EntropyComputation entropy_computation_criterion,
        ConjunctiveSetting conjunctive_setting)
        : job_manager(datapoint_ptrs, horn_constraints, solver) {
        _node_selection_criterion = node_selection_criterion;
        _entropy_computation_criterion = entropy_computation_criterion;
        _conjunctive_setting = conjunctive_setting;
    }

    /**
     * Creates a new complex job manager when a threshold is also passed.
     *
     * @param datapoint_ptrs A reference to the set of (pointers to) data points over which to work
     * @param horn_constraints A reference to the horn constraints over which to work
     * @param solver A reference to the Horn solver to use
     * @param threshold An unsigned int which serves as the threshold to cuts considered while splitting nodes wrt
     * numerical attributes
     * @param node_selection_criterion Node selection heuristic to be used while building the tree
     * @param entropy_computation_criterion Criterion for scoring a node/slice a la entropy
     */
    complex_job_manager(
        std::vector<datapoint<bool> *> &datapoint_ptrs,
        const std::vector<horn_constraint<bool>> &horn_constraints,
        horn_solver<bool> &solver,
        unsigned int threshold,
        NodeSelection node_selection_criterion,
        EntropyComputation entropy_computation_criterion,
        ConjunctiveSetting conjunctive_setting)
        : job_manager(datapoint_ptrs, horn_constraints, solver, threshold) {
        _node_selection_criterion = node_selection_criterion;
        _entropy_computation_criterion = entropy_computation_criterion;
        _conjunctive_setting = conjunctive_setting;
    }

    /**
     * Initializes _datapoint_ptrs_to_frac using purely the classified points in _datapoint_ptrs.
     */
    void initialize_datapoint_ptrs_to_frac();

    /**
     * Update _datapoint_ptrs_to_frac with randomly selected complete horn assignments
     */
    void update_datapoint_ptrs_to_frac_with_complete_horn_assignments();

    /**
     * Returns the next job.
     * If _node_selection_criterion is DEFAULT, calls the next_job() function of the super class
     *
     * @returns a unique pointer to the next job
     */
    std::unique_ptr<abstract_job> next_job() override;

    /**
     * Computes the entropy (with respect to the logarithm of 2) of a contiguous set of data
     * points.
     *
     * @param datapoint_ptrs Pointer to the data points
     * @param left_index The left bound of the set of data points
     * @param right_index The right bound of the set of data points
     *
     * @return the entropy of the given set of data points
     */
    double entropy(
        const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) override;

    /**
     * Computes the entropy (with respect to the logarithm of 2) of a contiguous set of data
     * points, weighted by the number of points classified in the set.
     *
     * @param datapoint_ptrs Pointer to the data points
     * @param left_index The left bound of the set of data points
     * @param right_index The right bound of the set of data points
     *
     * @return the entropy of the given set of data points
     */
    double weighted_entropy(
        const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) override;

    /**
     * Computes the number of classified points in a contiguous set of data points.
     *
     * @param datapoint_ptrs Pointer to the data points
     * @param left_index The left bound of the set of data points
     * @param right_index The right bound of the set of data points
     *
     * @return the number of classified points in a contiguous set of data points.
     */
    unsigned int num_classified_points(
        const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) override;

    /**
     * Computes the best split of a contiguous set of data points and returns the corresponding
     * split job. If no split (that allows progress) is possible, this function should throw
     * an exception.
     *
     * @param sl The slice of data points to be split
     *
     * @returns a unique pointer to the job created
     */
    std::unique_ptr<abstract_job> find_best_split(const slice &sl) override;

    void penalty(
        const slice &sl,
        std::size_t left_index,
        std::size_t cur_index,
        std::size_t right_index,
        int *left2right,
        int *right2left);
};
} // namespace horn_verification

