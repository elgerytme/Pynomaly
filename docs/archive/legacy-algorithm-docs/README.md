# Legacy Algorithm Documentation

## Archive Notice

**Date**: June 26, 2025  
**Action**: Algorithm Documentation Consolidation  

These files have been consolidated into a new unified algorithm reference structure to eliminate redundancy and improve navigation.

## Archived Files

### 1. `algorithms.md` (formerly `docs/guides/algorithms.md`)
- **Source**: Basic algorithm guide
- **Content**: 10 core algorithms with basic descriptions
- **Reason for Archive**: Merged into `docs/reference/algorithms/core-algorithms.md`

### 2. `algorithms-comprehensive.md` (formerly `docs/reference/algorithms-comprehensive.md`)  
- **Source**: Comprehensive algorithm reference
- **Content**: 100+ algorithms across multiple frameworks
- **Reason for Archive**: Split across specialized and experimental algorithm docs

### 3. `03-algorithm-options-functionality.md` (formerly `docs/comprehensive/03-algorithm-options-functionality.md`)
- **Source**: Detailed algorithm functionality guide
- **Content**: 45+ algorithms with extensive parameter documentation
- **Reason for Archive**: Merged into new structured algorithm docs

## New Structure

The content from these files has been reorganized into:

### **[`docs/reference/algorithms/`](../../reference/algorithms/)**
- **[README.md](../../reference/algorithms/README.md)** - Navigation and overview
- **[core-algorithms.md](../../reference/algorithms/core-algorithms.md)** - Essential algorithms (20+)
- **[specialized-algorithms.md](../../reference/algorithms/specialized-algorithms.md)** - Domain-specific algorithms
- **[experimental-algorithms.md](../../reference/algorithms/experimental-algorithms.md)** - Advanced/research methods
- **[algorithm-comparison.md](../../reference/algorithms/algorithm-comparison.md)** - Performance analysis

## Benefits of Consolidation

### ✅ **Eliminated Redundancy**
- No more triple-coverage of algorithms
- Single source of truth for each algorithm
- Consistent parameter documentation

### ✅ **Improved Navigation**
- Clear user journey paths
- Algorithms organized by use case and complexity
- Better cross-referencing

### ✅ **Enhanced Content Quality**
- Best content from all 3 sources merged
- Updated code examples using current API
- Performance guidance and comparison matrices

### ✅ **Reduced Maintenance**
- Single update point per algorithm
- Consistent format and structure
- Automated link validation

## Migration Guide

### For Internal References
- Update links from old paths to new structure
- Use `/docs/reference/algorithms/README.md` as main entry point
- Reference specific guides based on user needs

### For External Documentation
- Redirect links to appropriate new documentation
- Update bookmarks and documentation references
- Use new structure for better user experience

## Historical Context

These files represented the evolution of Pynomaly's algorithm documentation:
1. **Phase 1**: Basic guide with essential algorithms
2. **Phase 2**: Comprehensive reference with all algorithms  
3. **Phase 3**: Detailed functionality documentation

The new structure represents **Phase 4**: Organized, user-centric algorithm documentation designed for different user personas and use cases.

---

**For current algorithm documentation, visit**: [`docs/reference/algorithms/`](../../reference/algorithms/)