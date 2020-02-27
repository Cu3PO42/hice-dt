/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

#include <memory>
#include <limits>

namespace horn_verification {

class abstract_job;
class slice;
template<typename T> class datapoint;
class job_manager;
class complex_job_manager;

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
    using job_manager = job_manager;
    class split_index {
    public:
        double entropy = std::numeric_limits<double>::infinity();
        size_t index;

        split_index() = default;
        split_index(size_t index, const std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man, size_t _);

        split_index &assign_if_better(const split_index &other);
        double intrinsic_value_for_split(const std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man);

        constexpr bool is_possible() const;
    };

    int_split(size_t attribute, const slice &sl, job_manager &man);

    template<typename SplitT>
    typename SplitT::split_index find_index(std::vector<datapoint<bool> *> &datapoints);
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
    using job_manager = complex_job_manager;
    class split_index : public int_split::split_index {
        using base_split_index = int_split::split_index;
    public:
        split_index() = default;
        split_index(size_t index, const std::vector<datapoint<bool> *> &datapoints, const slice &sl, complex_job_manager &man, std::size_t total_classified_points);

        bool is_conjunctive;

        split_index &assign_if_better(split_index &&other);
    };

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
