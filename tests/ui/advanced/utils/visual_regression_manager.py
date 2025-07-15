"""
Visual Regression Manager for Advanced UI Testing

Provides comprehensive visual testing including:
- Screenshot comparison and diff analysis
- Baseline image management
- Cross-browser visual consistency
- Responsive design validation
- Visual change detection and reporting
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageChops
import numpy as np
from playwright.async_api import Page


class VisualRegressionManager:
    """
    Comprehensive visual regression testing manager
    """

    def __init__(self, baseline_dir: str = "test_artifacts/visual_baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.diff_dir = Path("test_artifacts/visual_diffs")
        self.diff_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.baseline_dir / "baseline_metadata.json"
        self.metadata = self._load_metadata()
        
        # Visual comparison thresholds
        self.thresholds = {
            "pixel_difference_percentage": 0.1,  # 0.1% pixel difference allowed
            "color_tolerance": 10,               # RGB color difference tolerance
            "ignore_antialiasing": True,         # Ignore minor antialiasing differences
            "ignore_alpha": False,               # Consider alpha channel in comparison
        }

    def _load_metadata(self) -> Dict[str, Any]:
        """Load baseline metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"baselines": {}, "version": "1.0"}

    def _save_metadata(self) -> None:
        """Save baseline metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Failed to save metadata: {e}")

    async def capture_baseline(self, page: Page, test_name: str, browser: str, viewport: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Capture baseline screenshot for visual regression testing
        
        Args:
            page: Playwright page object
            test_name: Name of the test/component
            browser: Browser name
            viewport: Viewport dimensions
            
        Returns:
            Baseline capture results
        """
        result = {
            "test_name": test_name,
            "browser": browser,
            "viewport": viewport,
            "baseline_path": None,
            "status": "failed"
        }
        
        try:
            # Generate baseline filename
            viewport_str = f"{viewport['width']}x{viewport['height']}" if viewport else "default"
            baseline_filename = f"{test_name}_{browser}_{viewport_str}.png"
            baseline_path = self.baseline_dir / baseline_filename
            
            # Set viewport if specified
            if viewport:
                await page.set_viewport_size(viewport)
                await page.wait_for_timeout(500)  # Wait for layout
            
            # Capture screenshot
            await page.screenshot(
                path=str(baseline_path),
                full_page=True,
                type="png"
            )
            
            # Calculate image hash for change detection
            image_hash = self._calculate_image_hash(baseline_path)
            
            # Update metadata
            baseline_key = f"{test_name}_{browser}_{viewport_str}"
            self.metadata["baselines"][baseline_key] = {
                "path": str(baseline_path),
                "hash": image_hash,
                "test_name": test_name,
                "browser": browser,
                "viewport": viewport,
                "created_at": self._get_timestamp(),
                "dimensions": self._get_image_dimensions(baseline_path)
            }
            
            self._save_metadata()
            
            result.update({
                "baseline_path": str(baseline_path),
                "image_hash": image_hash,
                "status": "success"
            })
            
        except Exception as e:
            result["error"] = str(e)
        
        return result

    async def compare_visual(self, page: Page, test_name: str, browser: str, viewport: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Compare current page against baseline for visual regression
        
        Args:
            page: Playwright page object
            test_name: Name of the test/component
            browser: Browser name
            viewport: Viewport dimensions
            
        Returns:
            Visual comparison results
        """
        result = {
            "test_name": test_name,
            "browser": browser,
            "viewport": viewport,
            "comparison_status": "failed",
            "difference_percentage": 0,
            "baseline_exists": False
        }
        
        try:
            # Generate comparison key
            viewport_str = f"{viewport['width']}x{viewport['height']}" if viewport else "default"
            baseline_key = f"{test_name}_{browser}_{viewport_str}"
            
            # Check if baseline exists
            if baseline_key not in self.metadata["baselines"]:
                result["error"] = "No baseline found. Run baseline capture first."
                return result
            
            result["baseline_exists"] = True
            baseline_info = self.metadata["baselines"][baseline_key]
            baseline_path = Path(baseline_info["path"])
            
            if not baseline_path.exists():
                result["error"] = f"Baseline file not found: {baseline_path}"
                return result
            
            # Set viewport if specified
            if viewport:
                await page.set_viewport_size(viewport)
                await page.wait_for_timeout(500)
            
            # Capture current screenshot
            current_filename = f"{test_name}_{browser}_{viewport_str}_current.png"
            current_path = self.diff_dir / current_filename
            
            await page.screenshot(
                path=str(current_path),
                full_page=True,
                type="png"
            )
            
            # Perform visual comparison
            comparison_result = self._compare_images(baseline_path, current_path, test_name, browser, viewport_str)
            result.update(comparison_result)
            
        except Exception as e:
            result["error"] = str(e)
        
        return result

    def _compare_images(self, baseline_path: Path, current_path: Path, test_name: str, browser: str, viewport_str: str) -> Dict[str, Any]:
        """
        Compare two images and generate difference analysis
        
        Args:
            baseline_path: Path to baseline image
            current_path: Path to current image
            test_name: Test name for diff filename
            browser: Browser name
            viewport_str: Viewport string
            
        Returns:
            Comparison results
        """
        try:
            # Load images
            baseline_img = Image.open(baseline_path).convert('RGBA')
            current_img = Image.open(current_path).convert('RGBA')
            
            # Ensure images have same dimensions
            if baseline_img.size != current_img.size:
                # Resize current image to match baseline
                current_img = current_img.resize(baseline_img.size, Image.Resampling.LANCZOS)
            
            # Calculate basic difference
            diff_img = ImageChops.difference(baseline_img, current_img)
            
            # Calculate difference percentage
            diff_percentage = self._calculate_difference_percentage(baseline_img, current_img)
            
            # Generate detailed analysis
            analysis = self._analyze_visual_differences(baseline_img, current_img, diff_img)
            
            # Create visual diff overlay
            diff_overlay_path = self._create_diff_overlay(
                baseline_img, current_img, diff_img, 
                test_name, browser, viewport_str
            )
            
            # Determine if comparison passes
            passes_threshold = diff_percentage <= self.thresholds["pixel_difference_percentage"]
            
            return {
                "comparison_status": "passed" if passes_threshold else "failed",
                "difference_percentage": diff_percentage,
                "threshold": self.thresholds["pixel_difference_percentage"],
                "baseline_path": str(baseline_path),
                "current_path": str(current_path),
                "diff_overlay_path": str(diff_overlay_path),
                "analysis": analysis,
                "dimensions": {
                    "baseline": baseline_img.size,
                    "current": current_img.size
                }
            }
            
        except Exception as e:
            return {
                "comparison_status": "error",
                "error": str(e)
            }

    def _calculate_difference_percentage(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculate percentage of different pixels"""
        try:
            # Convert to numpy arrays for efficient comparison
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            # Calculate differences with tolerance
            tolerance = self.thresholds["color_tolerance"]
            diff_mask = np.abs(arr1.astype(int) - arr2.astype(int)) > tolerance
            
            # Count different pixels (any channel difference counts)
            different_pixels = np.any(diff_mask, axis=-1)
            total_pixels = different_pixels.size
            diff_pixel_count = np.sum(different_pixels)
            
            return (diff_pixel_count / total_pixels) * 100
            
        except Exception:
            return 100.0  # Assume maximum difference on error

    def _analyze_visual_differences(self, baseline: Image.Image, current: Image.Image, diff: Image.Image) -> Dict[str, Any]:
        """Analyze visual differences in detail"""
        analysis = {
            "color_differences": {},
            "regional_differences": [],
            "major_changes": [],
            "minor_changes": []
        }
        
        try:
            # Convert to numpy for analysis
            baseline_arr = np.array(baseline)
            current_arr = np.array(current)
            diff_arr = np.array(diff)
            
            # Analyze color channel differences
            for i, channel in enumerate(['red', 'green', 'blue', 'alpha']):
                if i < baseline_arr.shape[2]:
                    baseline_channel = baseline_arr[:, :, i]
                    current_channel = current_arr[:, :, i]
                    
                    channel_diff = np.mean(np.abs(baseline_channel.astype(int) - current_channel.astype(int)))
                    analysis["color_differences"][channel] = float(channel_diff)
            
            # Find regions with significant changes
            gray_diff = np.mean(diff_arr, axis=2)
            significant_changes = gray_diff > 50  # Threshold for significant change
            
            if np.any(significant_changes):
                # Find contiguous regions of change
                regions = self._find_change_regions(significant_changes)
                analysis["regional_differences"] = regions
            
            # Classify changes as major or minor
            total_change = np.sum(gray_diff)
            if total_change > 1000000:  # Arbitrary threshold
                analysis["major_changes"].append({
                    "type": "layout_change",
                    "description": "Significant layout or content changes detected"
                })
            elif total_change > 100000:
                analysis["minor_changes"].append({
                    "type": "styling_change",
                    "description": "Minor styling or color changes detected"
                })
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis

    def _find_change_regions(self, change_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Find contiguous regions of visual changes"""
        regions = []
        
        try:
            # Simple region detection using connected components
            # This is a simplified version - in practice you might use scipy or opencv
            labeled = self._label_connected_components(change_mask)
            
            for label in np.unique(labeled):
                if label == 0:  # Skip background
                    continue
                
                region_mask = labeled == label
                coords = np.where(region_mask)
                
                if len(coords[0]) > 100:  # Only consider regions with significant size
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    
                    regions.append({
                        "bounds": {
                            "x": int(x_min),
                            "y": int(y_min),
                            "width": int(x_max - x_min),
                            "height": int(y_max - y_min)
                        },
                        "pixel_count": len(coords[0]),
                        "change_intensity": float(np.mean(change_mask[region_mask]))
                    })
            
        except Exception:
            pass
        
        return regions

    def _label_connected_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """Simple connected component labeling"""
        # Simplified implementation - in practice use scipy.ndimage.label
        labeled = np.zeros_like(binary_mask, dtype=int)
        current_label = 1
        
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if binary_mask[i, j] and labeled[i, j] == 0:
                    self._flood_fill(binary_mask, labeled, i, j, current_label)
                    current_label += 1
        
        return labeled

    def _flood_fill(self, mask: np.ndarray, labeled: np.ndarray, start_i: int, start_j: int, label: int) -> None:
        """Simple flood fill for connected component labeling"""
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1] or
                not mask[i, j] or labeled[i, j] != 0):
                continue
            
            labeled[i, j] = label
            
            # Add neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((i + di, j + dj))

    def _create_diff_overlay(self, baseline: Image.Image, current: Image.Image, diff: Image.Image, 
                           test_name: str, browser: str, viewport_str: str) -> Path:
        """Create visual diff overlay showing differences"""
        try:
            # Create side-by-side comparison
            width, height = baseline.size
            comparison_img = Image.new('RGB', (width * 3, height))
            
            # Paste baseline, current, and diff
            comparison_img.paste(baseline.convert('RGB'), (0, 0))
            comparison_img.paste(current.convert('RGB'), (width, 0))
            
            # Create enhanced diff visualization
            enhanced_diff = self._enhance_diff_visualization(diff)
            comparison_img.paste(enhanced_diff, (width * 2, 0))
            
            # Add labels
            draw = ImageDraw.Draw(comparison_img)
            try:
                # Try to use a better font, fallback to default
                from PIL import ImageFont
                font = ImageFont.load_default()
            except ImportError:
                font = None
            
            draw.text((10, 10), "Baseline", fill=(255, 255, 255), font=font)
            draw.text((width + 10, 10), "Current", fill=(255, 255, 255), font=font)
            draw.text((width * 2 + 10, 10), "Differences", fill=(255, 255, 255), font=font)
            
            # Save comparison image
            diff_filename = f"{test_name}_{browser}_{viewport_str}_diff.png"
            diff_path = self.diff_dir / diff_filename
            comparison_img.save(diff_path)
            
            return diff_path
            
        except Exception as e:
            print(f"Failed to create diff overlay: {e}")
            return self.diff_dir / "error_creating_diff.png"

    def _enhance_diff_visualization(self, diff: Image.Image) -> Image.Image:
        """Enhance difference visualization for better visibility"""
        try:
            # Convert to numpy for processing
            diff_arr = np.array(diff)
            
            # Enhance contrast of differences
            enhanced = np.zeros_like(diff_arr)
            
            # Convert to grayscale for processing
            gray_diff = np.mean(diff_arr[:, :, :3], axis=2)
            
            # Apply threshold and enhancement
            threshold = 10
            enhanced_mask = gray_diff > threshold
            
            # Color differences in red
            enhanced[enhanced_mask] = [255, 0, 0, 255]  # Red for differences
            enhanced[~enhanced_mask] = [50, 50, 50, 255]  # Dark gray for same areas
            
            return Image.fromarray(enhanced, 'RGBA').convert('RGB')
            
        except Exception:
            return diff.convert('RGB')

    def _calculate_image_hash(self, image_path: Path) -> str:
        """Calculate hash of image for change detection"""
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception:
            return ""

    def _get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions"""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return (0, 0)

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()

    async def run_cross_browser_visual_test(self, pages: Dict[str, Page], test_name: str, 
                                          viewports: List[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Run visual comparison across multiple browsers
        
        Args:
            pages: Dictionary of browser_name -> page
            test_name: Name of the test
            viewports: List of viewport dimensions to test
            
        Returns:
            Cross-browser visual test results
        """
        if viewports is None:
            viewports = [
                {"width": 1920, "height": 1080},
                {"width": 1366, "height": 768},
                {"width": 768, "height": 1024},
                {"width": 375, "height": 667}
            ]
        
        results = {
            "test_name": test_name,
            "browsers_tested": list(pages.keys()),
            "viewports_tested": viewports,
            "comparisons": [],
            "cross_browser_consistency": {},
            "overall_status": "pending"
        }
        
        try:
            # Capture screenshots for all browser/viewport combinations
            screenshots = {}
            
            for browser_name, page in pages.items():
                browser_screenshots = {}
                
                for viewport in viewports:
                    viewport_key = f"{viewport['width']}x{viewport['height']}"
                    
                    # Navigate to the page if needed
                    # await page.goto(url)  # URL should be passed or already loaded
                    
                    # Set viewport
                    await page.set_viewport_size(viewport)
                    await page.wait_for_timeout(1000)  # Wait for layout
                    
                    # Capture screenshot
                    screenshot_filename = f"{test_name}_{browser_name}_{viewport_key}.png"
                    screenshot_path = self.diff_dir / screenshot_filename
                    
                    await page.screenshot(
                        path=str(screenshot_path),
                        full_page=True,
                        type="png"
                    )
                    
                    browser_screenshots[viewport_key] = screenshot_path
                
                screenshots[browser_name] = browser_screenshots
            
            # Compare screenshots across browsers for each viewport
            for viewport in viewports:
                viewport_key = f"{viewport['width']}x{viewport['height']}"
                viewport_comparisons = []
                
                browsers = list(screenshots.keys())
                
                # Compare each browser with every other browser
                for i, browser1 in enumerate(browsers):
                    for browser2 in browsers[i+1:]:
                        img1_path = screenshots[browser1][viewport_key]
                        img2_path = screenshots[browser2][viewport_key]
                        
                        comparison = self._compare_images(
                            img1_path, img2_path, 
                            f"{test_name}_{viewport_key}", 
                            f"{browser1}_vs_{browser2}", 
                            viewport_key
                        )
                        
                        comparison.update({
                            "browser1": browser1,
                            "browser2": browser2,
                            "viewport": viewport
                        })
                        
                        viewport_comparisons.append(comparison)
                
                results["comparisons"].extend(viewport_comparisons)
            
            # Analyze cross-browser consistency
            results["cross_browser_consistency"] = self._analyze_cross_browser_consistency(results["comparisons"])
            
            # Determine overall status
            failed_comparisons = [c for c in results["comparisons"] if c["comparison_status"] == "failed"]
            results["overall_status"] = "failed" if failed_comparisons else "passed"
            results["failed_comparisons_count"] = len(failed_comparisons)
            
        except Exception as e:
            results["error"] = str(e)
            results["overall_status"] = "error"
        
        return results

    def _analyze_cross_browser_consistency(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cross-browser visual consistency"""
        analysis = {
            "average_difference": 0,
            "max_difference": 0,
            "problematic_combinations": [],
            "consistency_score": 0
        }
        
        try:
            if not comparisons:
                return analysis
            
            differences = [c.get("difference_percentage", 0) for c in comparisons]
            analysis["average_difference"] = sum(differences) / len(differences)
            analysis["max_difference"] = max(differences)
            
            # Find problematic browser combinations
            threshold = self.thresholds["pixel_difference_percentage"]
            for comparison in comparisons:
                if comparison.get("difference_percentage", 0) > threshold:
                    analysis["problematic_combinations"].append({
                        "browsers": f"{comparison['browser1']} vs {comparison['browser2']}",
                        "viewport": comparison["viewport"],
                        "difference": comparison["difference_percentage"]
                    })
            
            # Calculate consistency score (0-100)
            passed_comparisons = len([c for c in comparisons if c["comparison_status"] == "passed"])
            analysis["consistency_score"] = (passed_comparisons / len(comparisons)) * 100
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis

    def cleanup_old_artifacts(self, days_to_keep: int = 7) -> Dict[str, Any]:
        """Clean up old visual test artifacts"""
        cleanup_result = {
            "files_removed": 0,
            "space_freed_mb": 0,
            "errors": []
        }
        
        try:
            import time
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            
            for directory in [self.diff_dir, self.baseline_dir]:
                for file_path in directory.glob("*.png"):
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            cleanup_result["files_removed"] += 1
                            cleanup_result["space_freed_mb"] += file_size / (1024 * 1024)
                    except Exception as e:
                        cleanup_result["errors"].append(f"Failed to remove {file_path}: {e}")
            
        except Exception as e:
            cleanup_result["errors"].append(f"Cleanup failed: {e}")
        
        return cleanup_result

    def update_baseline(self, test_name: str, browser: str, viewport: Dict[str, int] = None) -> Dict[str, Any]:
        """Update baseline with current screenshot"""
        result = {"status": "failed", "message": ""}
        
        try:
            viewport_str = f"{viewport['width']}x{viewport['height']}" if viewport else "default"
            baseline_key = f"{test_name}_{browser}_{viewport_str}"
            
            # Find current screenshot
            current_filename = f"{test_name}_{browser}_{viewport_str}_current.png"
            current_path = self.diff_dir / current_filename
            
            if not current_path.exists():
                result["message"] = f"Current screenshot not found: {current_path}"
                return result
            
            # Move current to baseline
            baseline_filename = f"{test_name}_{browser}_{viewport_str}.png"
            baseline_path = self.baseline_dir / baseline_filename
            
            # Copy current to baseline location
            import shutil
            shutil.copy2(current_path, baseline_path)
            
            # Update metadata
            image_hash = self._calculate_image_hash(baseline_path)
            self.metadata["baselines"][baseline_key] = {
                "path": str(baseline_path),
                "hash": image_hash,
                "test_name": test_name,
                "browser": browser,
                "viewport": viewport,
                "updated_at": self._get_timestamp(),
                "dimensions": self._get_image_dimensions(baseline_path)
            }
            
            self._save_metadata()
            
            result.update({
                "status": "success",
                "message": f"Baseline updated for {baseline_key}",
                "baseline_path": str(baseline_path)
            })
            
        except Exception as e:
            result["message"] = str(e)
        
        return result

    def generate_visual_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive visual regression report"""
        report = f"""
# Visual Regression Test Report

## Test Overview
- **Test**: {test_results.get('test_name', 'Unknown')}
- **Overall Status**: {test_results.get('overall_status', 'Unknown')}
- **Browsers Tested**: {', '.join(test_results.get('browsers_tested', []))}

## Cross-Browser Consistency
"""
        
        consistency = test_results.get("cross_browser_consistency", {})
        if consistency:
            report += f"- **Consistency Score**: {consistency.get('consistency_score', 0):.1f}%\n"
            report += f"- **Average Difference**: {consistency.get('average_difference', 0):.2f}%\n"
            report += f"- **Maximum Difference**: {consistency.get('max_difference', 0):.2f}%\n"
            
            problematic = consistency.get("problematic_combinations", [])
            if problematic:
                report += "\n### Problematic Browser Combinations\n"
                for issue in problematic:
                    report += f"- {issue['browsers']} at {issue['viewport']['width']}x{issue['viewport']['height']}: {issue['difference']:.2f}% difference\n"
        
        # Add detailed comparison results
        comparisons = test_results.get("comparisons", [])
        if comparisons:
            report += f"\n## Detailed Comparisons ({len(comparisons)} total)\n"
            
            failed_comparisons = [c for c in comparisons if c["comparison_status"] == "failed"]
            if failed_comparisons:
                report += f"\n### Failed Comparisons ({len(failed_comparisons)})\n"
                for comp in failed_comparisons:
                    report += f"- {comp['browser1']} vs {comp['browser2']}: {comp['difference_percentage']:.2f}% difference\n"
            
            passed_comparisons = [c for c in comparisons if c["comparison_status"] == "passed"]
            if passed_comparisons:
                report += f"\n### Passed Comparisons ({len(passed_comparisons)})\n"
                report += "All visual comparisons within acceptable threshold.\n"
        
        # Add recommendations
        report += "\n## Recommendations\n"
        if test_results.get("overall_status") == "failed":
            report += "- Review failed visual comparisons and update baselines if changes are expected\n"
            report += "- Check for responsive design issues at different viewport sizes\n"
            report += "- Investigate browser-specific rendering differences\n"
        else:
            report += "- Visual consistency is maintained across tested browsers and viewports\n"
            report += "- Continue monitoring for visual regressions in future changes\n"
        
        return report