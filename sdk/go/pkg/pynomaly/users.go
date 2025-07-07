package pynomaly

import "context"

// UserManagementClient handles user and tenant management operations
type UserManagementClient struct {
	client *Client
}

// CreateUser creates a new user
func (u *UserManagementClient) CreateUser(ctx context.Context, req *CreateUserRequest) (*User, error) {
	var result User
	err := u.client.makeRequest(ctx, "POST", "/users", req, &result)
	return &result, err
}

// GetCurrentUser retrieves the current user profile
func (u *UserManagementClient) GetCurrentUser(ctx context.Context) (*User, error) {
	var result User
	err := u.client.makeRequest(ctx, "GET", "/users/me", nil, &result)
	return &result, err
}

// ListUsers returns a paginated list of users
func (u *UserManagementClient) ListUsers(ctx context.Context, opts *ListOptions) (*PaginatedResponse[User], error) {
	var result PaginatedResponse[User]
	err := u.client.makeRequest(ctx, "GET", "/users", nil, &result)
	return &result, err
}

// AddUserRole adds a role to a user
func (u *UserManagementClient) AddUserRole(ctx context.Context, userID string, role UserRole) (*User, error) {
	request := map[string]interface{}{
		"role": role,
	}

	var result User
	err := u.client.makeRequest(ctx, "POST", "/users/"+userID+"/roles", request, &result)
	return &result, err
}

// GetCurrentTenant retrieves the current tenant
func (u *UserManagementClient) GetCurrentTenant(ctx context.Context) (*Tenant, error) {
	var result Tenant
	err := u.client.makeRequest(ctx, "GET", "/tenants/current", nil, &result)
	return &result, err
}

// CreateTenant creates a new tenant
func (u *UserManagementClient) CreateTenant(ctx context.Context, req *CreateTenantRequest) (*Tenant, error) {
	var result Tenant
	err := u.client.makeRequest(ctx, "POST", "/tenants", req, &result)
	return &result, err
}

// Supporting types
type CreateTenantRequest struct {
	Name             string `json:"name"`
	Description      string `json:"description"`
	SubscriptionTier string `json:"subscription_tier"`
	MaxUsers         *int   `json:"max_users,omitempty"`
	MaxDetectors     *int   `json:"max_detectors,omitempty"`
	MaxDatasets      *int   `json:"max_datasets,omitempty"`
}