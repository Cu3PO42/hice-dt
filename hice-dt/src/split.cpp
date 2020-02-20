#include <functional>

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
    if (!split_possible) throw split_not_possible_error("This split is not possible");
    return std::make_unique<categorical_split_job>(*sl, attribute);
}

int_split::int_split(size_t attribute, std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man)
    : split(attribute, sl, man) {
    int tries = 0;
    double best_entropy = 1000000;
    double best_intrinsic_value = 0;

    // 1) Sort according to int attribute
    auto comparer = [attribute](const datapoint<bool> *const a, const datapoint<bool> *const b) {
        return a->_int_data[attribute] < b->_int_data[attribute];
    };
    std::sort(datapoints.begin() + sl._left_index, datapoints.begin() + sl._right_index + 1, comparer);

    // 2) Try all thresholds of current attribute
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
            if (!man._are_numerical_cuts_thresholded ||
                ((-1 * man._threshold <= datapoints[cur]->_int_data[attribute]) &&
                 (datapoints[cur]->_int_data[attribute] <= man._threshold))) {
                auto weighted_entropy_left = man.weighted_entropy(datapoints, sl._left_index, cur);
                auto weighted_entropy_right = man.weighted_entropy(datapoints, cur + 1, sl._right_index);
                auto total_weighted_entropy = weighted_entropy_left + weighted_entropy_right;

                if (!split_possible || total_weighted_entropy < best_entropy) {
                    split_possible = true;

                    best_entropy = total_weighted_entropy;
                    cut_index = cur;

                    // computation of the intrinsic value of the attribute
                    auto n1 = man.num_classified_points(datapoints, sl._left_index, cur);
                    auto n2 = man.num_classified_points(datapoints, cur + 1, sl._right_index);
                    auto n = n1 + n2;
                    best_intrinsic_value = calculate_intrinsic_value(n1, n) + calculate_intrinsic_value(n2, n);
                }
            }

            ++cur;
        }
    }
    if (split_possible) {
        // We have found the best split threshold for the given attribute
        // Now compute the information gain to optimize across different attributes
        double best_info_gain;
        if (man.num_classified_points(datapoints, sl._left_index, sl._right_index) == 0.0) {
            best_info_gain = man.entropy(datapoints, sl._left_index, sl._right_index);
        } else {
            best_info_gain =
                man.entropy(datapoints, sl._left_index, sl._right_index) -
                best_entropy /
                    man.num_classified_points(datapoints, sl._left_index, sl._right_index);
        }

        double interval = (datapoints[sl._right_index]->_int_data[attribute] -
                           datapoints[sl._left_index]->_int_data[attribute]) /
                          (datapoints[cut_index + 1]->_int_data[attribute] -
                           datapoints[cut_index]->_int_data[attribute]);

        assert(man.num_classified_points(datapoints, sl._left_index, sl._right_index) > 0);
        double threshCost = (interval < tries ? log2(interval) : log2(tries)) /
                            man.num_classified_points(datapoints, sl._left_index, sl._right_index);

        best_info_gain -= threshCost;
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
    if (!split_possible) throw split_not_possible_error("This split is not possible");
    return std::make_unique<int_split_job>(*sl, attribute, threshold);
}
