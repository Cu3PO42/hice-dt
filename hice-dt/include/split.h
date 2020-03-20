/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

#include <memory>
#include <limits>
#include <variant>
#include <vector>

namespace horn_verification {

class abstract_job;
class slice;
template<typename T> class datapoint;
class job_manager;
class complex_job_manager;
class index_list;

class split {
protected:
    split() = default;
    split(std::size_t attribute, const slice &sl, job_manager &man);

    bool split_possible;
    double gain_ratio;
    std::size_t attribute;
    const slice *sl;
    job_manager *man;

    static double calculate_intrinsic_value(double fraction, double total);

public:
    virtual std::unique_ptr<abstract_job> make_job() const = 0;

    bool is_possible() const;

    bool operator<(const split &other) const;
};

class int_split : public split {
protected:
    using manager = job_manager;
    class split_index {
    public:
        double entropy = std::numeric_limits<double>::infinity();
        size_t index;

        split_index() = default;
        #define SPLIT_INDEX_ARGS(T) size_t left_index, size_t right_index, const std::vector<datapoint<bool> *> &datapoints, const slice &sl, typename T::manager &man, size_t total_classified_points
        #define SPLIT_INDEX_ARGS_VARS left_index, right_index, datapoints, sl, man, total_classified_points
        split_index(SPLIT_INDEX_ARGS(int_split));

        double intrinsic_value_for_split(const std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man);

        constexpr bool is_possible() const;

        constexpr bool operator<(const split_index &rhs) const;
    };
    using base_split = split_index;
    using all_splits = std::variant<split_index>;

    int_split(size_t attribute, const slice &sl, job_manager &man);

    template<typename SplitT>
    typename SplitT::all_splits find_index(std::vector<datapoint<bool> *> &datapoints);

    template<typename SplitT, typename ... Splits>
    void construct_all(std::variant<Splits...> &cur_best, SPLIT_INDEX_ARGS(SplitT));

    template<typename SplitT, typename SplitList>
    void assign_better(typename SplitT::all_splits &cur_best, SPLIT_INDEX_ARGS(SplitT));

    double calculate_info_gain(std::vector<datapoint<bool> *> &datapoints, double entropy, std::size_t total_classified_poins);

    int threshold;
    double intrinsic_value;
    size_t cut_index;

public:
    int_split() = default;
    int_split(size_t attribute, std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man);

    int_split &assign_if_better(int_split &&other);
    std::unique_ptr<abstract_job> make_job() const override;
};

class complex_int_split : public int_split {
private:
    using manager = complex_job_manager;
    class complex_split_index_base : public int_split::split_index {
    public:
        using split_index::split_index;

        void compute_entropy(SPLIT_INDEX_ARGS(complex_int_split), const index_list &left_child_indices, const index_list &right_child_indices);
        constexpr bool operator<(const complex_split_index_base &other) const;

        bool is_conjunctive;
    };
    class split_index_le : public complex_split_index_base {
    public:
        split_index_le() = default;
        split_index_le(SPLIT_INDEX_ARGS(complex_int_split));
    };
    class split_index_eq : public complex_split_index_base {
    public:
        split_index_eq() = default;
        split_index_eq(SPLIT_INDEX_ARGS(complex_int_split));
    };
    using base_split = complex_split_index_base;
    using all_splits = std::variant<split_index_le, split_index_eq>;

    friend class int_split;
    double calculate_info_gain(std::vector<datapoint<bool> *> &datapoints, double entropy, std::size_t total_classified_poins);

    bool is_conjunctive;

public:
    complex_int_split() = default;
    complex_int_split(size_t attribute, std::vector<datapoint<bool> *> &datapoints, const slice &sl, complex_job_manager &man);

    complex_int_split &assign_if_better(complex_int_split &&other);
};

class cat_split : public split {
public:
    cat_split() = default;
    cat_split(size_t attribute, std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man);
    cat_split &assign_if_better(cat_split &&other);

    std::unique_ptr<abstract_job> make_job() const override;
};

}
