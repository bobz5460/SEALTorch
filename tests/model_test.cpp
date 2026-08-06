#include "model.h"

#include <cassert>
#include <stdexcept>

int main() {
    sealtorch::Model model;
    model.add({ 2, 3, {{1, 0}, {0, 1}, {1, 1}}, {0, 0, 0} }).activation();
    model.add({ 3, 1, {{1, 1, 1}}, {0} });
    assert(model.input_size() == 2);
    assert(model.output_size() == 1);
    assert(model.activation_count() == 1);
    model.activation();
    bool rejected = false;
    try { model.activation(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
