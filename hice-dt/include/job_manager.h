/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

// C++ includes
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// C includes
#include <cassert>
#include <cmath>

// Project includes
#include "datapoint.h"
#include "error.h"
#include "horn_constraint.h"
#include "horn_solver.h"
#include "job.h"
#include "slice.h"

namespace horn_verification {

/**
 * Enum to decide the heuristic for selecting the next node while constructing the tree
 */
enum NodeSelection {
    BFS = 0,              // Selects node in a Breadth-first order
    DFS,                  // Selects node in a Depth-first order
    RANDOM,               // Selects a random node
    MAX_ENTROPY,          // Selects the node which has the maximum entropy
    MAX_WEIGHTED_ENTROPY, // Selects the node which has the maximum entropy weighted by the number of classified points
                          // in the node
    MIN_ENTROPY,          // Selects the node which has the minimum entropy
    MIN_WEIGHTED_ENTROPY  // Selects the node which has the minimum entropy wieghted by the number of classified points
                          // in the node
};
/**
 * Enum to select if one prefers conjunctive splits (split carves out a sub-node which includes only negative points or
 * unclassified points) over non-conjunctive splits
 */
enum ConjunctiveSetting { NOPREFERENCEFORCONJUNCTS = 0, PREFERENCEFORCONJUNCTS };

/**
 * Enum to decide the heuristic for computing the goodness/badness score (a la entropy) for a set of data points.
 */
enum EntropyComputation {
    DEFAULT_ENTROPY = 0, // Ignores horn constraints and computes the entropy using only the positively and negatively
                         // classified data points
    PENALTY,             // Adds a linear penalty term based on the number of horn constraints involving data points
                         // outside the current set
    HORN_ASSIGNMENTS,    // Estimates the positive/negative distribution of the unclassified points and considers that
                         // when computing the entropy
};

/**
 * Abstract job manager.
 *
 * A job manager implements heuristics for decising
 * <ul>
 *   <li> when to split and when to turn a \ref slice into a leaf node; and</li>
 *   <li> if a \ref slice is to be split, how to split it.</li>
 * </ul>
 *
 * A job manager has to implement the following three methods:
 * <ul>
 *   <li> add_slice(const slice &) (preferably also add_slice(slice &&)):
 *        This method adds a new \ref slice to the job manager.</li>
 *   <li> next_job(): This method returns the job the manager wants to have
 *        performed next. A job can be to splt a node (i.e., either a
 *        \ref categorical_split_job or an \ref int_split_job, depending on
 *        which type of attribute the learner should split on) or to create a
 *        leaf node (i.e., a \ref leaf_creation_job). In order to facilitate
 *        polymorphism and be exception safe, the manager returns a \p unique_ptr.</li>
 *   <li> has_jobs(): This method returns whether there are jobts that the
 *        learner needs to process.</li>
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
class job_manager {
protected:
    /// The slices that need to be processed
    std::list<slice> _slices;

    /// A reference to the set of (pointers to) data points
    std::vector<datapoint<bool> *> &_datapoint_ptrs;

    /// A reference to the horn constraints
    const std::vector<horn_constraint<bool>> &_horn_constraints;

    /// The solver for Horn clauses
    horn_solver<bool> &_horn_solver;

    /// Threshold which bounds the numerical cuts considered while constructing the tree
    int _threshold;
    bool _are_numerical_cuts_thresholded;

    bool _is_first_split = true;

public:
    /**
     * Creates a new job manager.
     *
     * @param datapoint_ptrs A reference to the set of (pointers to) data points over which to work
     * @param horn_constraints A reference to the horn constraints over which to work
     * @param solver A reference to the Horn solver to use
     */
    job_manager(
        std::vector<datapoint<bool> *> &datapoint_ptrs,
        const std::vector<horn_constraint<bool>> &horn_constraints,
        horn_solver<bool> &solver)
        : _datapoint_ptrs(datapoint_ptrs),
          _horn_constraints(horn_constraints),
          _horn_solver(solver),
          _are_numerical_cuts_thresholded(false),
          _threshold(0) {
    }

    /**
     * Creates a new job manager when a threshold is also passed.
     *
     * @param datapoint_ptrs A reference to the set of (pointers to) data points over which to work
     * @param horn_constraints A reference to the horn constraints over which to work
     * @param solver A reference to the Horn solver to use
     * @param threshold An unsigned int which serves as the threshold to cuts considered while splitting nodes wrt
     * numerical attributes
     */
    job_manager(
        std::vector<datapoint<bool> *> &datapoint_ptrs,
        const std::vector<horn_constraint<bool>> &horn_constraints,
        horn_solver<bool> &solver,
        int threshold)
        : _datapoint_ptrs(datapoint_ptrs),
          _horn_constraints(horn_constraints),
          _horn_solver(solver),
          _threshold(threshold),
          _are_numerical_cuts_thresholded(true) {
    }

    /**
     * Adds a new slice to the manager.
     *
     * @param sl the slice to add
     */
    void add_slice(const slice &sl) { _slices.push_back(sl); }

    /**
     * Adds a new slice to the manager using move semantics.
     *
     * @param sl the slice to add
     */
    void add_slice(slice &&sl) { _slices.push_back(std::move(sl)); }

    /**
     * Checks whether the job manager has jobs that need to be processed.
     *
     * @returns whether the job manager has jobs that need to be processed
     */
    [[nodiscard]] inline bool has_jobs() const { return !_slices.empty(); }

    /**
     * Returns the next job. If no job exists (i.e., calling learner::empty() returns \c true),
     * the bahavior is undefined.
     *
     * @returns a unique pointer to the next job
     */
    virtual std::unique_ptr<abstract_job> next_job() = 0;

protected:
    /**
     * Checks whether a slice can be turned into a leaf node. If so, this method also
     * determines the label of the leaf node and which unlabeled data points need to
     * be labaled positively and negatively, respectively, in order to satisfy the
     * horn constraints.
     *
     * @param sl The slice to check
     * @param label If the slice can be turned into a leaf, this parameter is used to return the label
     *              of the leaf node
     * @param positive_ptrs If the slice can be turned into a leaf, this paramater is used to return
     *                      a set of (pointers to) unlabaled data points which have to be labeled positively
     * @param negative_ptrs If the slice can be turned into a leaf, this paramater is used to return
     *                      a set of (pointers to) unlabaled data points which have to be labeled negatively
     *
     * @return whether this slice can be turned into a leaf node
     */
    bool is_leaf(
        const slice &sl,
        bool &label,
        std::unordered_set<datapoint<bool> *> &positive_ptrs,
        std::unordered_set<datapoint<bool> *> &negative_ptrs);

    template <class T>
    void output_state(
        const std::unordered_set<datapoint<T> *> &positive_ptrs,
        const std::unordered_set<datapoint<T> *> &negative_ptrs,
        const std::vector<horn_constraint<T>> &horn_constraints,
        const std::string &headline,
        std::ostream &out) {
        // Headline
        out << headline << std::endl;

        // Positive data points
        out << "Positive data points (" << positive_ptrs.size() << "): " << std::endl;
        for (const auto &dp : positive_ptrs) {
            out << *dp << std::endl;
        }

        // Negative data points
        out << "Negative data points (" << negative_ptrs.size() << "): " << std::endl;
        for (const auto &dp : negative_ptrs) {
            out << *dp << std::endl;
        }

        // Horn constraints
        out << "Horn constraints (" << horn_constraints.size() << "):" << std::endl;
        for (const auto &clause : horn_constraints) {
            for (const auto &dp : clause._premises) {
                out << "(" << *dp << ") ";
            }

            out << " ==>  ";
            if (clause._conclusion) {
                out << "(" << *clause._conclusion << ")";
            } else {
                out << "(null)";
            }
            out << std::endl;
        }
    }

    /**
     * Computes the best split of a contiguous set of data points and returns the corresponding
     * split job. If no split (that allows progress) is possible, this function should throw
     * an exception.
     *
     * @param sl The slice of data points to be split
     *
     * @returns a unique pointer to the job created
     */
    virtual std::unique_ptr<abstract_job> find_best_split(const slice &sl) = 0;

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
    virtual double entropy(
        const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) = 0;

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
    virtual double weighted_entropy(
        const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) = 0;

    /**
     * Computes the number of classified points in a contiguous set of data points.
     *
     * @param datapoint_ptrs Pointer to the data points
     * @param left_index The left bound of the set of data points
     * @param right_index The right bound of the set of data points
     *
     * @return the number of classified points in a contiguous set of data points.
     */
    virtual unsigned int num_classified_points(
        const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) = 0;

    /**
     * Returns if an unclassified point is present in a contiguous set of data points.
     *
     * @param datapoint_ptrs Pointer to the data points
     * @param left_index The left bound of the set of data points
     * @param right_index The right bound of the set of data points
     *
     * @return If an unclassified point is present in the contiguous set of data points.
     */
    static bool unclassified_points_present(
        const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index);

    /**
     * Returns if a positively labeled point is present in a contiguous set of data points.
     *
     * @param datapoint_ptrs Pointer to the data points
     * @param left_index The left bound of the set of data points
     * @param right_index The right bound of the set of data points
     *
     * @return If a positively labeled point is present in the contiguous set of data points.
     */
    static bool positive_points_present(
        const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index);

    /**
     * Returns number of points with the given classification in a contiguous set of data points.
     *
     * @param datapoint_ptrs Pointer to the data points
     * @param left_index The left bound of the set of data points
     * @param right_index The right bound of the set of data points
     * @param classification The classification to count the number of points for
     *
     * @return If a positively labeled point is present in the contiguous set of data points.
     */
    static int num_points_with_classification(
        const std::vector<datapoint<bool> *> &datapoint_ptrs,
        std::size_t left_index,
        std::size_t right_index,
        bool classification);
};

} // namespace horn_verification