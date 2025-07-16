# IAM Policy Examples & Least-Privilege Recommendations

## Quick Start

Implement IAM policies with a focus on least-privilege principles. These configurations aim to restrict access to only the necessary resources and actions based on user roles.

### Policy Examples

#### Admin Policy
Provides full access across all resources.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "*",
            "Resource": "*"
        }
    ]
}
```

#### Read-Only Policy
Allows reading data but denies modification or deletion.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:Get*",
                "s3:List*"
            ],
            "Resource": "arn:aws:s3:::example-bucket/*"
        }
    ]
}
```

### Least-Privilege Recommendations

1. **Role-Based Access Control (RBAC):** Utilize roles to segregate duties and minimize access.
   - **Administrator:** Full access for managing resources.
   - **Developer:** Limited to deploying and managing their own applications.
   - **Analyst:** Access to data-related operations.

2. **Follow the Principle of Least Privilege (PoLP):**
   - Regularly audit and refine permissions.
   - Review permissions attached to roles and users periodically.
   - Use conditions with IAM policies to constrain access based on factors like source IP, request time, etc.

3. **Use IAM Groups:** Group users with similar access needs for easier management.
   - Assign policies to groups rather than individual users.
   - Ensure that groups are named intuitively based on their function (e.g., `Developers`, `Analysts`).

4. **Enable Multi-Factor Authentication (MFA):**
   - Require MFA for sensitive API calls and console logins.

5. **Implement Logging and Monitoring:**
   - Track IAM activities with AWS CloudTrail.
   - Integrate with security information and event management (SIEM) tools for deeper insights.

## Further Reading
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [IAM JSON Policy Elements: Action, Resource, and Principal](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements.html)

---

Please refer to the full [Security Guide](../deployment/security.md) for extended documentation on security best practices.
