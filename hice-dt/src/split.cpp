#include <functional>

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
    return total == 0 ? 0 : -(fraction / total) * log2(fraction/total);
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
  find_index<int_split>(datapoints);
}

template<typename SplitT>
typename SplitT::split_index int_split::find_index(std::vector<datapoint<bool> *> &datapoints) {
    auto &man = *this->man;
    auto &sl = *this->sl;

    int tries = 0;
    typename SplitT::split_index best_index;

    // 1) Sort according to int attribute
    auto attribute = this->attribute;
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
                best_index.assign_if_better(typename SplitT::split_index(cur, datapoints, sl, static_cast<typename SplitT::manager &>(man), total_classified_points));
            }

            ++cur;
        }
    }

    split_possible = best_index.is_possible();
    cut_index = best_index.index;

    if (split_possible) {
        // We have found the best split threshold for the given attribute
        // Now compute the information gain to optimize across different attributes
        double best_info_gain = static_cast<SplitT *>(this)->calculate_info_gain(datapoints, best_index.entropy, total_classified_points);

        double interval = (datapoints[sl._right_index]->_int_data[attribute] -
                           datapoints[sl._left_index]->_int_data[attribute]) /
                          (datapoints[cut_index + 1]->_int_data[attribute] -
                           datapoints[cut_index]->_int_data[attribute]);

        // TODO: why is a case with 0 classified points not reasonable?
        assert(total_classified_points > 0);
        double threshCost = (interval < tries ? log2(interval) : log2(tries)) /
                            total_classified_points;

        best_info_gain -= threshCost;

        intrinsic_value = best_index.intrinsic_value_for_split(datapoints, sl, man);
        gain_ratio =
          intrinsic_value > 0
          ? best_info_gain / intrinsic_value
          : best_info_gain;
        threshold = datapoints[cut_index]->_int_data[attribute];
    }

    return best_index;
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

int_split::int_split(size_t attribute, const slice &sl, job_manager &man) : split(attribute, sl, man)
{}
double int_split::calculate_info_gain(
    std::vector<datapoint<bool> *> &datapoints, double entropy, std::size_t total_classified_points) {
    if (total_classified_points == 0.0) {
      return man->entropy(datapoints, sl->_left_index, sl->_right_index);
    } else {
      return
          man->entropy(datapoints, sl->_left_index, sl->_right_index) -
          entropy / total_classified_points;
    }
}

int_split::split_index::split_index(size_t index, const std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man, std::size_t _)
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
    return entropy < std::numeric_limits<double>::infinity();
}

complex_int_split::complex_int_split(
    size_t attribute,
    std::vector<datapoint<bool> *> &datapoints,
    const slice &sl,
    complex_job_manager &man)
    : int_split(attribute, sl, man) {

    auto best_split = find_index<complex_int_split>(datapoints);
    is_conjunctive = man._conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS && best_split.is_conjunctive;
}

complex_int_split &complex_int_split::assign_if_better(complex_int_split &&other) {
    if (!other.split_possible) return *this;
    std::size_t conj_preferred =
        static_cast<complex_job_manager *>(other.man)->_conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS
        ? 1
        : 0;
    auto lhs = std::make_tuple(is_conjunctive * conj_preferred, intrinsic_value > 0, gain_ratio, -threshold);
    auto rhs = std::make_tuple(other.is_conjunctive * conj_preferred, other.intrinsic_value > 0, other.gain_ratio, -other.threshold);
    if (lhs <= rhs) {
        *this = other;
    }
    return *this;
}
double complex_int_split::calculate_info_gain(
    std::vector<datapoint<bool> *> &datapoints, double entropy, std::size_t total_classified_poins) {
    return man->entropy(datapoints, sl->_left_index, sl->_right_index) - entropy;
}

complex_int_split::split_index::split_index(
    size_t index, const std::vector<datapoint<bool> *> &datapoints, const slice &sl, complex_job_manager &man, std::size_t total_classified_points)
    : base_split_index(index, datapoints, sl, man, total_classified_points)
    , is_conjunctive(
        // One of the sub node consists purely of negative or unclassified points.
        // Consider this as a prospective candidate for a conjunctive split
        man._conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS &&
        (!man.positive_points_present(datapoints, sl._left_index, index) ||
        !man.positive_points_present(datapoints, index + 1, sl._right_index)) ) {
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
complex_int_split::split_index &
complex_int_split::split_index::assign_if_better(complex_int_split::split_index &&other) {
    if (std::tie(is_conjunctive, entropy) < std::tie(other.is_conjunctive, other.entropy)) {
      *this = other;
    }
    return *this;
}
