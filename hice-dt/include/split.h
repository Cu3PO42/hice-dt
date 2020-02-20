/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

#include <memory>

#include "job.h"
#include "job_manager.h"

namespace horn_verification {

class split {
protected:
    split() = default;
    split(std::size_t attribute, const slice &sl, job_manager &man);

    bool split_possible;
    double gain_ratio;
    std::size_t attribute;
    const slice *sl;
    job_manager *man;

    double calculate_intrinsic_value(double fraction, double total);
public:
    virtual std::unique_ptr<abstract_job> make_job() const = 0;

    bool is_possible() const;

    bool operator<(const split &other) const;
};

class int_split : public split {
    int threshold;
    size_t cut_index;

public:
    int_split() = default;
    int_split(size_t attribute, std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man);

    int_split &assign_if_better(int_split &&other);
    std::unique_ptr<abstract_job> make_job() const override;
};

class cat_split : public split {
public:
    cat_split() = default;
    cat_split(size_t attribute, std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man);
    cat_split &assign_if_better(cat_split &&other);

    std::unique_ptr<abstract_job> make_job() const override;
};

}
