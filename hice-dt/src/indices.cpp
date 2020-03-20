#include "indices.h"

namespace horn_verification {
index_list::index_list(std::size_t left, std::size_t right)
    : member { .elem = { left, right } }
    , is_external(false) {
}

index_list::index_list(std::initializer_list<index_pair> indices)
    : member { .elem = indices.size() == 1 ? *indices.begin() : index_pair { 0, 0 } }
    , is_external(true) {
    if (indices.size() != 1) {
        member.external.ptr = new index_pair[indices.size()];
        member.external.size = indices.size();

        std::size_t i = 0;
        for (auto &pair : indices) {
            member.external.ptr[i] = pair;
            ++i;
        }
    }
}

index_list::~index_list() {
    if (is_external) {
        delete[] member.external.ptr;
    }
}

const index_pair *index_list::begin() const noexcept {
    if (is_external) {
        return member.external.ptr;
    }
    return &member.elem;
}

const index_pair *index_list::end() const noexcept {
    if (is_external) {
        return member.external.ptr + member.external.size;
    }
    return &member.elem + 1;
}
}