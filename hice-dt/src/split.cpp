#include <functional>

#include "job.h"
#include "job_manager.h"
#include "complex_job_manager.h"
#include "split.h"

using namespace horn_verification;

// =======================================
// meta programming helpers
// =======================================

template<typename... Ts>
struct tlist;

template<typename T, typename... Ts>
struct tlist<T, Ts...> {
    using head = T;
    using tail = tlist<Ts...>;
    static constexpr bool has = true;
};
template<>
struct tlist<> {
    static constexpr bool has = false;
};

template<typename V, typename Base, std::size_t cur = 0>
Base *get_as(V *variant) {
    if (auto it = std::get_if<cur>(variant)) {
        return static_cast<Base *>(it);
    }
    if constexpr (cur + 1 < std::variant_size_v<V>) {
        return get_as<V, Base, cur + 1>(variant);
    }
    // is unreachable
    assert(false);
    return nullptr;
}


// =======================================
// base split functions
// =======================================

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

// =======================================
// cat_split
// =======================================
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

    auto total_classified_points = man.num_classified_points(datapoints, { sl._left_index, sl._right_index });

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
            total_weighted_entropy += man.weighted_entropy(datapoints, { cur_left, cur_right });

            auto classified_points_current_category = man.num_classified_points(datapoints, { cur_left, cur_right });
            total_intrinsic_value += calculate_intrinsic_value(classified_points_current_category, total_classified_points);

            cur_left = cur_right + 1;
            cur_right = cur_left;
        }
    }

    if (split_possible) {
        double info_gain;
        if (total_classified_points == 0) {
            info_gain = man.entropy(datapoints, { sl._left_index, sl._right_index });
        } else {
            info_gain =
                man.entropy(datapoints, { sl._left_index, sl._right_index }) -
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

// =======================================
// int_split
// =======================================
int_split::int_split(size_t attribute, std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man)
    : split(attribute, sl, man) {
  find_index<int_split>(datapoints);
}

template<typename SplitT>
typename SplitT::all_splits int_split::find_index(std::vector<datapoint<bool> *> &datapoints) {
    auto &man = *this->man;
    auto &sl = *this->sl;

    int tries = 0;
    typename SplitT::all_splits best_index;

    // 1) Sort according to int attribute
    auto attribute = this->attribute;
    auto comparer = [attribute](const datapoint<bool> *const a, const datapoint<bool> *const b) {
        return a->_int_data[attribute] < b->_int_data[attribute];
    };
    std::sort(datapoints.begin() + sl._left_index, datapoints.begin() + sl._right_index + 1, comparer);

    // 2) Try all thresholds of current attribute
    auto total_classified_points = man.num_classified_points(datapoints, { sl._left_index, sl._right_index });
    auto cur = sl._left_index;
    while (cur < sl._right_index) {
        auto left = cur;
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
                construct_all<SplitT>(best_index, left, cur, datapoints, sl, static_cast<typename SplitT::manager &>(man), total_classified_points);
            }

            ++cur;
        }
    }

    auto best_index_base = *get_as<typename SplitT::all_splits, typename SplitT::base_split>(&best_index);
    split_possible = best_index_base.is_possible();
    cut_index = best_index_base.index;

    if (split_possible) {
        // FIXME: some computations here may need to be adapted to different split_index types
        // We have found the best split threshold for the given attribute
        // Now compute the information gain to optimize across different attributes
        // NOTE: this computation is ok for both <= and ==
        double best_info_gain = static_cast<SplitT *>(this)->calculate_info_gain(datapoints, best_index_base.entropy, total_classified_points);

        // NOTE: this really only makes sence for the <= split, not for ==
        double interval = (datapoints[sl._right_index]->_int_data[attribute] -
                           datapoints[sl._left_index]->_int_data[attribute]) /
                          (datapoints[cut_index + 1]->_int_data[attribute] -
                           datapoints[cut_index]->_int_data[attribute]);

        // TODO: why is a case with 0 classified points not reasonable?
        assert(total_classified_points > 0);
        // Tries is the number of distinct values that were tested for splitting
        double threshCost = log2(std::min(interval, tries)) /
                            total_classified_points;

        best_info_gain -= threshCost;

        intrinsic_value = best_index_base.intrinsic_value_for_split(datapoints, sl, man);
        gain_ratio =
          intrinsic_value > 0
          ? best_info_gain / intrinsic_value
          : best_info_gain;
        threshold = datapoints[cut_index]->_int_data[attribute];
    }

    return best_index;
}


template<typename SplitT, typename ... Splits>
void int_split::construct_all(std::variant<Splits...> &cur_best, SPLIT_INDEX_ARGS(SplitT)) {
    assign_better<SplitT, tlist<Splits...>>(cur_best, SPLIT_INDEX_ARGS_VARS);
}

template<typename SplitT, typename SplitList>
void int_split::assign_better(typename SplitT::all_splits &cur_best, SPLIT_INDEX_ARGS(SplitT)) {
    if constexpr (SplitList::has) {
        auto best = get_as<typename SplitT::all_splits, typename SplitT::base_split>(&cur_best);
        typename SplitList::head next(SPLIT_INDEX_ARGS_VARS);
        if (*best < next) cur_best = std::move(next);
        assign_better<SplitT, typename SplitList::tail>(cur_best, SPLIT_INDEX_ARGS_VARS);
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

int_split::int_split(size_t attribute, const slice &sl, job_manager &man) : split(attribute, sl, man)
{}

double int_split::calculate_info_gain(
    std::vector<datapoint<bool> *> &datapoints, double entropy, std::size_t total_classified_points) {
    if (total_classified_points == 0.0) {
      return man->entropy(datapoints, { sl->_left_index, sl->_right_index });
    } else {
      return
          man->entropy(datapoints, { sl->_left_index, sl->_right_index }) -
          entropy / total_classified_points;
    }
}

// =======================================
// int_split::split_index
// =======================================
int_split::split_index::split_index(SPLIT_INDEX_ARGS(int_split))
    : index(right_index) {
    auto weighted_entropy_left = man.weighted_entropy(datapoints, { sl._left_index, index });
    auto weighted_entropy_right = man.weighted_entropy(datapoints, { index + 1, sl._right_index });
    entropy = weighted_entropy_left + weighted_entropy_right;
}

double int_split::split_index::intrinsic_value_for_split(
    const std::vector<datapoint<bool> *> &datapoints, const slice &sl, job_manager &man) {
    auto n1 = man.num_classified_points(datapoints, { sl._left_index, index });
    auto n2 = man.num_classified_points(datapoints, { index + 1, sl._right_index });
    return calculate_intrinsic_value(n1, n1+n2) + calculate_intrinsic_value(n2, n1+n2);
}

constexpr bool int_split::split_index::is_possible() const {
    return entropy < std::numeric_limits<double>::infinity();
}

constexpr bool int_split::split_index::operator<(const split_index &rhs) const {
    // We want to minimize entropy
    return entropy > rhs.entropy;
}

// =======================================
// complex_int_split
// =======================================
complex_int_split::complex_int_split(
    size_t attribute,
    std::vector<datapoint<bool> *> &datapoints,
    const slice &sl,
    complex_job_manager &man)
    : int_split(attribute, sl, man) {

    auto best_split = find_index<complex_int_split>(datapoints);
    is_conjunctive = man._conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS && get_as<all_splits, base_split>(&best_split)->is_conjunctive;
    split_index_le test;
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
    return man->entropy(datapoints, { sl->_left_index, sl->_right_index }) - entropy;
}

// =======================================
// complext_int_split::split_index
// =======================================
constexpr bool complex_int_split::complex_split_index_base::operator<(const complex_int_split::complex_split_index_base &other) const {
    return std::tie(is_conjunctive, entropy) >= std::tie(other.is_conjunctive, other.entropy);
}

void complex_int_split::complex_split_index_base::compute_entropy(
    SPLIT_INDEX_ARGS(complex_int_split),
    const index_list &left_child_indices,
    const index_list &right_child_indices) {
    is_conjunctive =
        // One of the sub node consists purely of negative or unclassified points.
        // Consider this as a prospective candidate for a conjunctive split
        man._conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS &&
        (!man.positive_points_present(datapoints, left_child_indices) ||
        !man.positive_points_present(datapoints, right_child_indices))
    ;
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
        double negative_left = man.num_points_with_classification(datapoints, left_child_indices, false);
        double positive_left = man.num_points_with_classification(datapoints, left_child_indices, true);
        double negative_right =
            man.num_points_with_classification(datapoints, right_child_indices, false);
        double positive_right =
            man.num_points_with_classification(datapoints, right_child_indices, true);
        double total_classified_points = negative_left + positive_left + negative_right + positive_right;

        auto negative_left_rel = negative_left == 0 ? 0 : negative_left / (negative_left + positive_left);
        auto positive_left_rel = positive_left == 0 ? 0 : positive_left / (negative_left + positive_left);
        auto negative_right_rel = negative_right == 0 ? 0 : negative_right / (negative_right + positive_right);
        auto positive_right_rel = positive_right == 0 ? 0 : positive_right / (negative_right + positive_right);

        double penaltyVal = (1 - negative_left_rel * positive_right_rel) * left2right + (1 - negative_right_rel * positive_left_rel) * right2left;
        penaltyVal = 2 * penaltyVal / (2 * (left2right + right2left) + total_classified_points);
        entropy += penaltyVal;
    }
}

complex_int_split::split_index_le::split_index_le(SPLIT_INDEX_ARGS(complex_int_split))
    : complex_split_index_base(SPLIT_INDEX_ARGS_VARS) {
    index_list left_child_indices(sl._left_index, index);
    index_list right_child_indices(index + 1, sl._right_index);
    compute_entropy(SPLIT_INDEX_ARGS_VARS, left_child_indices, right_child_indices);
}

complex_int_split::split_index_eq::split_index_eq(SPLIT_INDEX_ARGS(complex_int_split))
    : complex_split_index_base() {
    index_list left_child_indices(left_index, right_index);
    index_list right_child_indices {{ sl._left_index, left_index - 1}, { right_index + 1, sl._right_index }};
    compute_entropy(SPLIT_INDEX_ARGS_VARS, left_child_indices, right_child_indices);
}
