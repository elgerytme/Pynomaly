# Pynomaly Database ERD (Entity-Relationship Diagram)

This document describes the ERD for the Pynomaly user management and metrics database schema.

## Tables and Relationships

### Users
- `id`: Primary Key
- `name`: User's full name
- `email`: User's email address
- `created_at`: Timestamp when the user was created

### Roles
- `id`: Primary Key
- `name`: Role name

### UserRoles
- `user_id`: Foreign Key referencing Users(id)
- `role_id`: Foreign Key referencing Roles(id)
- Composite Primary Key: (user_id, role_id)

### Tenants
- `id`: Primary Key
- `name`: Tenant's name
- `created_at`: Timestamp when the tenant was created

### Metrics
- `id`: Primary Key
- `type`: Type of metric
- `value`: Value of the metric
- `collected_at`: Timestamp when the metric was collected

## Indices
- Unique constraints on `email` in Users
- Foreign Key constraints for `user_id` and `role_id` in UserRoles

## Permission Matrix Integration

The permission matrix defines access levels:
- **Super Admin**: Full platform access
- **Tenant Admin**: Manage own tenant
- **Data Scientist**: Manage own datasets and models
- **Analyst**: Run detections and create reports
- **Viewer**: Read-only access

Each role has specific access to actions like `CREATE`, `READ`, `UPDATE`, and `DELETE` on resources such as `tenants`, `users`, `datasets`, etc.
