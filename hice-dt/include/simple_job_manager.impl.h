/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

#include "simple_job_manager.h"

namespace horn_verification {

template<typename CatT, typename IntT, typename ManT>
std::unique_ptr<abstract_job> simple_job_manager::find_best_split_base(const slice &sl) {
    assert(sl._left_index <= sl._right_index && sl._right_index < _datapoint_ptrs.size());

    //
    // Process categorical attributes
    //
    CatT best_cat_split;

    for (std::size_t attribute = 0; attribute < _datapoint_ptrs[sl._left_index]->_categorical_data.size();
         ++attribute) {
        best_cat_split.assign_if_better(CatT(attribute, _datapoint_ptrs, sl, *static_cast<ManT *>(this)));
    }

    //
    // Process integer attributes
    //
    IntT best_int_split;

    for (std::size_t attribute = 0; attribute < _datapoint_ptrs[sl._left_index]->_int_data.size(); ++attribute) {
        best_int_split.assign_if_better(IntT(attribute, _datapoint_ptrs, sl, *static_cast<ManT *>(this)));
    }

    //
    // Return best split
    //
    assert(best_int_split.is_possible() || best_cat_split.is_possible());
    assert(best_int_split.is_possible());
    return best_int_split.make_job();
    return std::max<split>(best_int_split, best_cat_split).make_job();
}

}