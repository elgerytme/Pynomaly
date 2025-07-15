# GitHub Issue Management Rules

## Purpose
This document establishes standardized rules for managing GitHub issues throughout their lifecycle to ensure proper tracking, transparency, and synchronization with repository changes.

## Issue Status Management Rules

### Rule 1: Issue State Synchronization
**Requirement**: All GitHub issues must accurately reflect their current development status at all times.

**Implementation**:
- Issues must be updated within 24 hours of any status change
- Status changes must be reflected in both GitHub issue labels and issue state
- Automated status updates are preferred when possible

### Rule 2: Completed Issues Must Be Closed
**Requirement**: When work on an issue is completed and all acceptance criteria are met, the issue must be marked as resolved/closed in GitHub.

**Implementation**:
- Close the issue immediately after final commit is pushed
- Add a closing comment referencing the commit(s) that resolved the issue
- Use GitHub's "Fixes #issue-number" syntax in commit messages for automatic closure
- Example: `git commit -m "feat: implement data validation engine\n\nFixes #140"`

### Rule 3: In-Progress Issues Must Be Marked
**Requirement**: Issues being actively worked on must be clearly marked as in-progress.

**Implementation**:
- Add "in-progress" label when work begins
- Assign the issue to the developer working on it
- Update the issue with progress comments at least weekly
- Move issue to "In Progress" column in project boards if using GitHub Projects

### Rule 4: Active Work Visibility
**Requirement**: Issues being actively worked on must be clearly visible to the team.

**Implementation**:
- Use GitHub's "In Progress" project status
- Add regular progress updates in issue comments
- Link related pull requests to the issue
- Update issue description with current implementation status

## Repository Synchronization Rules

### Rule 5: Commit-Issue Linkage
**Requirement**: All commits related to issue resolution must be properly linked to the issue.

**Implementation**:
- Use conventional commit messages with issue references
- Include issue number in commit messages: `feat(#140): add validation engine`
- Link commits to issues using GitHub's closing keywords:
  - `Closes #issue-number`
  - `Fixes #issue-number`
  - `Resolves #issue-number`

### Rule 6: Completed Work Must Be Committed and Synced
**Requirement**: When an issue is marked as completed, all related repository changes must be committed and synchronized.

**Implementation**:
- Ensure all changes are committed to the repository
- Push commits to the remote repository
- Create and merge pull requests for completed work
- Verify all tests pass before marking issue as complete
- Tag releases when appropriate

### Rule 7: Pull Request Integration
**Requirement**: All significant issue resolutions must go through pull request review.

**Implementation**:
- Create pull requests for all issue resolutions
- Link pull requests to issues using GitHub's linking syntax
- Ensure pull request descriptions reference the issue
- Require review approval before merging
- Automatically close issues when pull requests are merged

## Issue Lifecycle Management

### Rule 8: Issue Creation Standards
**Requirement**: New issues must follow standardized creation practices.

**Implementation**:
- Use issue templates when available
- Include clear acceptance criteria
- Add appropriate labels and milestones
- Assign priority levels
- Link to related issues or epics

### Rule 9: Issue Update Requirements
**Requirement**: Issues must be kept current with regular updates.

**Implementation**:
- Update progress at least weekly for active issues
- Add comments for significant developments
- Update labels and assignees as needed
- Close stale issues after appropriate notice period

### Rule 10: Issue Resolution Verification
**Requirement**: Issue completion must be verified before closure.

**Implementation**:
- Verify all acceptance criteria are met
- Run full test suite
- Conduct code review if applicable
- Confirm deployment success for features
- Document any remaining technical debt

## Automation and Tooling

### Rule 11: Automated Status Updates
**Requirement**: Use automation to maintain issue status accuracy.

**Implementation**:
- Configure GitHub Actions for automatic issue updates
- Use commit message parsing for automatic issue closure
- Set up branch protection rules requiring issue links
- Implement automated testing before issue closure
- Use CLI commands for batch issue management: `gh issue list --label "in-progress" --state open`
- Regular audits using: `gh issue list --state open --limit 100` to identify stale issues
- Automated closure of completed issues with documentation: `gh issue close <number> --comment "completion details"`

### Rule 12: Integration with Development Workflow
**Requirement**: Issue management must integrate seamlessly with development workflow.

**Implementation**:
- Use GitHub Projects for kanban-style tracking
- Integrate with CI/CD pipelines
- Configure notifications for status changes
- Link issues to project milestones and releases

## Compliance and Monitoring

### Rule 13: Regular Audits
**Requirement**: Conduct regular audits of issue management compliance.

**Implementation**:
- Weekly review of open issues
- Monthly audit of closed issues
- Quarterly process improvement reviews
- Annual workflow optimization

### Rule 14: Metrics and Reporting
**Requirement**: Track metrics for issue management effectiveness.

**Implementation**:
- Monitor issue resolution time
- Track issue reopening rates
- Measure commit-to-issue linking compliance
- Report on development velocity

## Implementation Checklist

- [ ] Set up GitHub issue templates
- [ ] Configure automated issue labeling
- [ ] Implement commit message standards
- [ ] Set up pull request templates with issue linking
- [ ] Configure GitHub Actions for automated updates
- [ ] Train team on new workflow
- [ ] Establish monitoring and audit processes
- [ ] Document exception handling procedures
- [ ] Implement CLI-based issue management automation
- [ ] Create scripts for bulk issue status updates
- [ ] Establish regular issue audit schedule using CLI tools

## Exception Handling

### Emergency Fixes
- Critical hotfixes may bypass normal PR process
- Still requires issue creation and linking
- Must be retroactively reviewed within 24 hours

### External Dependencies
- Issues blocked by external factors must be clearly marked
- Regular updates required on blocking status
- Alternative solutions should be explored

### Legacy Issues
- Existing issues must be brought into compliance within 30 days
- Historical issues may be closed with appropriate documentation
- Migration plan required for large backlogs

---

**Last Updated**: 2025-01-15
**Version**: 1.0
**Review Schedule**: Quarterly