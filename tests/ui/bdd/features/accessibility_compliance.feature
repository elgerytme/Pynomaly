Feature: Accessibility Compliance and Inclusive Design
  As a user with accessibility needs
  I want the Pynomaly web application to be fully accessible
  So that I can use all features regardless of my abilities or assistive technology

  Background:
    Given the Pynomaly web application is running
    And accessibility features are enabled

  Scenario: Screen Reader Navigation
    Given I am using a screen reader
    When I navigate to the homepage
    Then all content should be accessible via screen reader
    And page landmarks should be properly labeled
    And heading structure should be logical
    And images should have appropriate alt text
    And interactive elements should have accessible names

  Scenario: Keyboard-Only Navigation
    Given I am using only keyboard navigation
    When I navigate through the application
    Then all interactive elements should be keyboard accessible
    And tab order should be logical and predictable
    And focus indicators should be clearly visible
    And skip links should be available for main content
    And keyboard shortcuts should work as expected

  Scenario: High Contrast Mode Support
    Given I need high contrast for visual accessibility
    When I enable high contrast mode
    Then all text should have sufficient color contrast
    And UI elements should remain distinguishable
    And focus indicators should be enhanced
    And interactive elements should be clearly defined
    And visual hierarchy should be maintained

  Scenario: Text Scaling and Zoom Support
    Given I need larger text for readability
    When I scale text to 200% size
    Then all content should remain accessible
    And layout should not break with larger text
    And interactive elements should remain usable
    And horizontal scrolling should not be required
    And information should not be lost

  Scenario: Reduced Motion Preferences
    Given I prefer reduced motion for vestibular safety
    When I enable reduced motion preferences
    Then animations should be minimized or disabled
    And auto-playing content should be paused
    And transitions should respect motion preferences
    And parallax effects should be disabled
    And the interface should remain fully functional

  Scenario: Form Accessibility
    Given I am filling out a form
    When I interact with form elements
    Then all form fields should have proper labels
    And required fields should be clearly indicated
    And error messages should be announced to screen readers
    And field descriptions should be associated with inputs
    And form validation should be accessible

  Scenario: Data Table Accessibility
    Given I am viewing a data table
    When I navigate through the table data
    Then table headers should be properly marked up
    And column/row relationships should be clear
    And table caption should describe the data
    And sorting controls should be accessible
    And table navigation should work with assistive technology

  Scenario: Modal Dialog Accessibility
    Given I encounter a modal dialog
    When the dialog opens
    Then focus should move to the dialog
    And background content should be hidden from screen readers
    And the dialog should have proper role and labeling
    And I should be able to close it with keyboard
    And focus should return to the trigger element when closed

  Scenario: Interactive Charts and Visualizations
    Given I am viewing data visualizations
    When I interact with charts and graphs
    Then alternative text descriptions should be provided
    And data should be available in tabular format
    And chart interactions should work with keyboard
    And sonification or other alternatives should be available
    And chart data should be accessible to screen readers

  Scenario: Error Handling Accessibility
    Given an error occurs in the application
    When the error is displayed
    Then error messages should have proper ARIA roles
    And errors should be announced to screen readers
    And error context should be provided
    And recovery instructions should be clear
    And error states should be keyboard accessible

  Scenario: Language and Internationalization
    Given the application supports multiple languages
    When content language changes
    Then proper lang attributes should be set
    And text direction should be handled correctly
    And cultural considerations should be respected
    And date/time formats should be localized
    And currency and number formats should be appropriate

  Scenario: Cognitive Accessibility Support
    Given I have cognitive accessibility needs
    When I use the application
    Then navigation should be consistent across pages
    And complex interactions should have clear instructions
    And timeout warnings should be provided
    And help documentation should be easily accessible
    And the interface should minimize cognitive load

  Scenario: Mobile Accessibility
    Given I am using a mobile device with assistive technology
    When I navigate the mobile interface
    Then touch targets should be appropriately sized
    And gestures should have alternatives
    And screen reader navigation should work smoothly
    And pinch-to-zoom should be supported
    And orientation changes should not break functionality

  Scenario: Compliance Validation
    Given the application is deployed
    When accessibility audits are performed
    Then WCAG 2.1 AA compliance should be achieved
    And automated accessibility tests should pass
    And manual accessibility testing should pass
    And user testing with disabled users should be positive
    And accessibility documentation should be complete