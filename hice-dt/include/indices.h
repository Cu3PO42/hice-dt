#pragma once

#include <utility>

namespace horn_verification {

struct index_pair {
    std::size_t left, right;
};

class index_list {
    union {
    public:
        struct {
            index_pair *ptr;
            std::size_t size;
        } external;
        index_pair elem;
    } member;

    bool is_external;

public:
    index_list(std::size_t left, std::size_t right);
    index_list(std::initializer_list<index_pair> indices);
    ~index_list();

    const index_pair *begin() const noexcept;
    const index_pair *end() const noexcept;
};
}