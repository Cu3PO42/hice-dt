/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

namespace horn_verification {
template<typename T>
bool less(T lhs, T rhs) {
    return lhs < rhs;
}

template<typename T>
bool greater(T lhs, T rhs) {
    return lhs > rhs;
}
};