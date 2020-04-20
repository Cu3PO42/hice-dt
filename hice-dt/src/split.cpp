#include <functional>
#include <iostream>

#include "job.h"
#include "job_manager.h"
#include "complex_job_manager.h"
#include "split.h"

using namespace horn_verification;

split::split(std::size_t attribute, const slice &sl, job_manager &man)
    : attribute(attribute)
    , sl(&sl)
    , man(&man) {
}
double split::calculate_intrinsic_value(double fraction, double total) {
    return -(fraction / total) * log2(fraction/total);
}
bool split::is_possible() const { return split_possible; }

bool split::operator<(const split &other) const {
    return std::tie(split_possible, gain_ratio) < std::tie(other.split_possible, other.gain_ratio);
}

cat_split::cat_split(std::size_t attribute, std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man)
    : split(attribute, sl, man)
{
    // 1) Sort according to categorical attribute
    auto comparer = [attribute](auto lhs, auto rhs) {
        return lhs->_categorical_data[attribute] < rhs->_categorical_data[attribute];
    };
    std::sort(
        datapoints.begin() + sl._left_index,
        datapoints.begin() + sl._right_index + 1,
        comparer
    );

    // 2) sum all weighted entropies
    double total_weighted_entropy = 0;
    double total_intrinsic_value = 0;
    auto cur_left = sl._left_index;
    auto cur_right = cur_left;
    split_possible = true;

    auto total_classified_points = man.num_classified_points(datapoints, sl._left_index, sl._right_index);

    while (cur_right <= sl._right_index) {
        auto cur_category = datapoints[cur_left]->_categorical_data[attribute];

        while (cur_right + 1 <= sl._right_index &&
               cur_category == datapoints[cur_right + 1]->_categorical_data[attribute]) {
            ++cur_right;
        }

        // If only one category, skip attribute
        if (cur_left == sl._left_index && cur_right == sl._right_index) {
            split_possible = false;
            break;
        } else {
            total_weighted_entropy += man.weighted_entropy(datapoints, cur_left, cur_right);

            auto classified_points_current_category = man.num_classified_points(datapoints, cur_left, cur_right);
            total_intrinsic_value += calculate_intrinsic_value(classified_points_current_category, total_classified_points);

            cur_left = cur_right + 1;
            cur_right = cur_left;
        }
    }

    if (split_possible) {
        double info_gain;
        if (total_classified_points == 0) {
            info_gain = man.entropy(datapoints, sl._left_index, sl._right_index);
        } else {
            info_gain =
                man.entropy(datapoints, sl._left_index, sl._right_index) -
                total_weighted_entropy / total_classified_points;
        }

        assert(total_intrinsic_value > 0.0);
        gain_ratio = info_gain / total_intrinsic_value;
    }
}

cat_split &cat_split::assign_if_better(cat_split &&other) {
    if (!other.split_possible || gain_ratio >= other.gain_ratio) return *this;
    return *this = other;
}

std::unique_ptr<abstract_job> cat_split::make_job() const {
    if (!split_possible) throw split_not_possible_error("This cat split is not possible");
    return std::make_unique<categorical_split_job>(*sl, attribute);
}

int_split::int_split(size_t attribute, std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man)
    : split(attribute, sl, man) {
    int tries = 0;
    split_index best_index;

    // 1) Sort according to int attribute
    auto comparer = [attribute](const datapoint<bool> *const a, const datapoint<bool> *const b) {
        return a->_int_data[attribute] < b->_int_data[attribute];
    };
    std::sort(datapoints.begin() + sl._left_index, datapoints.begin() + sl._right_index + 1, comparer);

    // 2) Try all thresholds of current attribute
    auto total_classified_points = man.num_classified_points(datapoints, sl._left_index, sl._right_index);
    auto cur = sl._left_index;
    while (cur < sl._right_index) {
        // Skip to riight most entry with the same value
        while (cur + 1 <= sl._right_index &&
               datapoints[cur + 1]->_int_data[attribute] == datapoints[cur]->_int_data[attribute]) {
            ++cur;
        }

        // Split is possible
        if (cur < sl._right_index) {
            tries++;

            // if cuts have been thresholded, check that a split at the current value of the numerical attribute is
            // allowed
            auto attribute_value =datapoints[cur]->_int_data[attribute];
            if (!man._are_numerical_cuts_thresholded ||
                ((-man._threshold <= attribute_value) && (attribute_value <= man._threshold))) {
                best_index.assign_if_better(split_index(cur, datapoints, sl, man));
            }

            ++cur;
        }
    }

    split_possible = best_index.is_possible();
    cut_index = best_index.index;

    if (split_possible) {
        // We have found the best split threshold for the given attribute
        // Now compute the information gain to optimize across different attributes
        double best_info_gain;
        if (total_classified_points == 0.0) {
            best_info_gain = man.entropy(datapoints, sl._left_index, sl._right_index);
        } else {
            best_info_gain =
                man.entropy(datapoints, sl._left_index, sl._right_index) -
                best_index.entropy / total_classified_points;
        }

        double interval = (datapoints[sl._right_index]->_int_data[attribute] -
                           datapoints[sl._left_index]->_int_data[attribute]) /
                          (datapoints[cut_index + 1]->_int_data[attribute] -
                           datapoints[cut_index]->_int_data[attribute]);

        assert(total_classified_points > 0);
        double threshCost = (interval < tries ? log2(interval) : log2(tries)) /
                            total_classified_points;

        best_info_gain -= threshCost;

        double best_intrinsic_value = best_index.intrinsic_value_for_split(datapoints, sl, man);
        assert(best_intrinsic_value > 0.0);
        gain_ratio = best_info_gain / best_intrinsic_value;
        threshold = datapoints[cut_index]->_int_data[attribute];
    }
}

int_split &int_split::assign_if_better(int_split &&other) {
    // TODO: replace this >= by > to get larger cuts
    // TODO: ask Daniel about this weird comparison
    if (!other.split_possible ||
        std::make_tuple(gain_ratio, std::abs(other.threshold)) >=
        std::make_tuple(other.gain_ratio, std::abs(threshold))
        ) return *this;
    return *this = other;
}
std::unique_ptr<abstract_job> int_split::make_job() const {
    char underlying_val = *reinterpret_cast<const char*>(&split_possible);
    if (underlying_val != 0 && underlying_val != 1) {
        std::cerr << "ups. " << (int)underlying_val << std::endl;
        exit(-1);
    }
    assert(is_possible());
    std::cerr << "helo i bims" << std::endl;
    if (!is_possible()) throw split_not_possible_error("Foo. This int split is not possible");
    return std::make_unique<int_split_job>(*sl, attribute, threshold);
}

int_split::int_split(size_t attribute, const slice &sl, job_manager &man) : split(attribute, sl, man)
{}

int_split::split_index::split_index(size_t index, const std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man)
    : index(index) {
    auto weighted_entropy_left = man.weighted_entropy(datapoints, sl._left_index, index);
    auto weighted_entropy_right = man.weighted_entropy(datapoints, index + 1, sl._right_index);
    entropy = weighted_entropy_left + weighted_entropy_right;
}

int_split::split_index &int_split::split_index::assign_if_better(const int_split::split_index &other) {
    if (other.entropy < entropy) *this = other;
    return *this;
}
double int_split::split_index::intrinsic_value_for_split(
    const std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man) {
    auto n1 = man.num_classified_points(datapoints, sl._left_index, index);
    auto n2 = man.num_classified_points(datapoints, index + 1, sl._right_index);
    return calculate_intrinsic_value(n1, n1+n2) + calculate_intrinsic_value(n2, n1+n2);
}
constexpr bool int_split::split_index::is_possible() const {
    return (entropy < std::numeric_limits<double>::infinity()) != 0 ? 1 : 0;
}

complex_int_split::complex_int_split(
    size_t attribute,
    std::vector<datapoint<bool> *> &datapoints,
    const slice &sl,
    complex_job_manager &man)
    : int_split(attribute, sl, man) {
    for (std::size_t attribute = 0; attribute < datapoints[sl._left_index]->_int_data.size(); ++attribute) {
        int tries = 0;
        split_index best_normal_split, best_conj_split;

        // 1) Sort according to int attribute
        auto comparer = [attribute](const datapoint<bool> *const a, const datapoint<bool> *const b) {
            return a->_int_data[attribute] < b->_int_data[attribute];
        };
        std::sort(
            datapoints.begin() + sl._left_index, datapoints.begin() + sl._right_index + 1, comparer);

        // 2) Try all thresholds of current attribute
        auto total_classified_points = man.num_classified_points(datapoints, sl._left_index, sl._right_index);
        auto cur = sl._left_index;
        while (cur < sl._right_index) {
            // Skip to riight most entry with the same value
            while (cur + 1 <= sl._right_index &&
                   datapoints[cur + 1]->_int_data[attribute] == datapoints[cur]->_int_data[attribute]) {
                ++cur;
            }

            // Split is possible
            if (cur < sl._right_index) {
                tries++;

                // if cuts have been thresholded, check that a split at the current value of the numerical attribute
                // is allowed
                if (!man._are_numerical_cuts_thresholded ||
                    ((-1 * man._threshold <= datapoints[cur]->_int_data[attribute]) &&
                     (datapoints[cur]->_int_data[attribute] <= man._threshold))) {
                    split_index current_split(cur, datapoints, sl, man);

                    // If the learner prefers conjunctive splits
                    if (man._conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS) {
                        // Check if a conjunctive split is possible
                        if (!man.positive_points_present(datapoints, sl._left_index, cur) ||
                            !man.positive_points_present(datapoints, cur + 1, sl._right_index)) {
                            // One of the sub node consists purely of negative or unclassified points.
                            // Consider this as a prospective candidate for a conjunctive split
                            best_conj_split.assign_if_better(current_split);
                        }
                    }

                    best_normal_split.assign_if_better(current_split);
                }

                ++cur;
            }
        }

        is_conjuncive = man._conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS && best_conj_split.is_possible();
        auto &best_split = is_conjuncive ? best_conj_split : best_normal_split;

        split_possible = best_split.is_possible();
        cut_index = best_split.index;

        if (split_possible) {
            // We have found the best split threshold for the given attribute
            // Now compute the information gain to optimize across different attributes
            double best_info_gain;
            best_info_gain =
                man.entropy(datapoints, sl._left_index, sl._right_index) - best_split.entropy;

            double interval = (datapoints[sl._right_index]->_int_data[attribute] -
                               datapoints[sl._left_index]->_int_data[attribute]) /
                              (datapoints[best_split.index + 1]->_int_data[attribute] -
                               datapoints[best_split.index]->_int_data[attribute]);

            assert(total_classified_points > 0);
            double threshCost = (interval < tries ? log2(interval) : log2(tries)) /
                                total_classified_points;

            best_info_gain -= threshCost;
            double best_intrinsic_value = best_split.intrinsic_value_for_split(datapoints, sl, man);
            has_positive_intrinsic_value = best_intrinsic_value > 0;

            double best_gain_ratio_for_given_attribute =
                has_positive_intrinsic_value
                ? best_info_gain / best_intrinsic_value
                : best_info_gain;

            gain_ratio = best_gain_ratio_for_given_attribute;
            threshold = datapoints[best_split.index]->_int_data[attribute];
        }
    }
}

complex_int_split &complex_int_split::assign_if_better(complex_int_split &&other) {
    if (!other.split_possible) return *this;
    std::size_t conj_preferred =
        static_cast<complex_job_manager *>(other.man)->_conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS
        ? 1
        : 0;
    auto lhs = std::make_tuple(is_conjuncive * conj_preferred, has_positive_intrinsic_value, gain_ratio, -threshold);
    auto rhs = std::make_tuple(other.is_conjuncive * conj_preferred, other.has_positive_intrinsic_value, other.gain_ratio, -other.threshold);
    if (lhs <= rhs) {
        *this = other;
    }
    return *this;
}

complex_int_split::complex_split_index::complex_split_index(
    size_t index, const std::vector<datapoint<bool> *> &datapoints, const slice &sl, complex_job_manager &man, std::size_t total_classified_points)
    : split_index(index, datapoints, sl, man) {

    if (total_classified_points == 0) {
        entropy = 0.0;
    } else {
        entropy /= total_classified_points;
    }

    // Add a penalty based on the number of implications in the horn constraints that are cut by the
    // current split.
    if (man._entropy_computation_criterion == EntropyComputation::PENALTY) {
        // number of implications in the horn constraints cut by the current split.
        int left2right, right2left;
        std::tie(left2right, right2left) = man.penalty(sl._left_index, index, sl._right_index);
        double negative_left = man.num_points_with_classification(datapoints, sl._left_index, index, false);
        double positive_left = man.num_points_with_classification(datapoints, sl._left_index, index, true);
        double negative_right =
            man.num_points_with_classification(datapoints, index + 1, sl._right_index, false);
        double positive_right =
            man.num_points_with_classification(datapoints, index + 1, sl._right_index, true);
        double total_classified_points = negative_left + positive_left + negative_right + positive_right;

        // FIXME: in the following negative_left is used after being assigned; was the original value meant?
        // same for negative_right
        negative_left = negative_left == 0 ? 0 : negative_left / (negative_left + positive_left);
        positive_left = positive_left == 0 ? 0 : positive_left / (negative_left + positive_left);
        negative_right = negative_right == 0 ? 0 : negative_right / (negative_right + positive_right);
        positive_right = positive_right == 0 ? 0 : positive_right / (negative_right + positive_right);

        double penaltyVal = (1 - negative_left * positive_right) * left2right + (1 - negative_right * positive_left) * right2left;
        penaltyVal = 2 * penaltyVal / (2 * (left2right + right2left) + total_classified_points);
        entropy += penaltyVal;
    }
}
