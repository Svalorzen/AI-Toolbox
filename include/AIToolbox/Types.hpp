#ifndef AI_TOOLBOX_TYPES_HEADER_FILE
#define AI_TOOLBOX_TYPES_HEADER_FILE

#include <boost/multi_array.hpp>

namespace AIToolbox {
    using Table3D = boost::multi_array<double, 3>;
    using Table2D = boost::multi_array<double, 2>;
}

#endif
