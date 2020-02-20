/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

#include "job_manager.h"

namespace horn_verification {

/**
 * Implements a simple job manager using entropy as the split heuristic.
 */
class simple_job_manager : public job_manager {
public:
    /**
     * Inherit constructors from base class.
     */
     using job_manager::job_manager;

    /**
     * @see job_manager::next_job()
     *
     * This implementation is simplistic in that it retrieves slices in a breadth-first order
     * and uses a simple entropy measure to split slices.
     */
     std::unique_ptr<abstract_job> next_job() override;

protected:
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

    /**
     * Computes the best split of a contiguous set of data points and returns the corresponding
     * split job. If no split (that allows progress) is possible, this function should throw
     * an exception.
     *
     * This is a shared base function for the simple and complex job manager. It allows
     * customization of the attribute selection by specifying subclasses of the split class
     * to bo used for categorical and integer attributes.
     *
     * @param sl The slice of data points to be split
     *
     * @returns a unique pointer to the job created
     */
    template<typename CatT, typename IntT, typename ManT>
    std::unique_ptr<abstract_job> find_best_split_base(const slice &sl);

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

    friend class int_split;
    friend class complex_int_split;
    friend class cat_split;
};

}; // End namespace horn_verification
