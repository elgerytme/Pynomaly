/**
 * User Management client for user and tenant operations
 */

import { AxiosInstance } from 'axios';
import { 
  User, 
  CreateUserRequest, 
  Tenant, 
  UserRole,
  ListOptions, 
  PaginatedResponse 
} from '../types';
import { PynomalyError, AuthenticationError, AuthorizationError } from '../errors';

export class UserManagementClient {
  constructor(private httpClient: AxiosInstance) {}

  // User Management

  /**
   * List all users
   */
  async listUsers(options: ListOptions = {}): Promise<PaginatedResponse<User>> {
    try {
      const response = await this.httpClient.get('/users', { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to list users', 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Get user by ID
   */
  async getUser(userId: string): Promise<User> {
    try {
      const response = await this.httpClient.get(`/users/${userId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Get current user profile
   */
  async getCurrentUser(): Promise<User> {
    try {
      const response = await this.httpClient.get('/users/me');
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new AuthenticationError('Failed to get current user profile');
    }
  }

  /**
   * Create new user
   */
  async createUser(request: CreateUserRequest): Promise<User> {
    try {
      const response = await this.httpClient.post('/users', request);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to create user', 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Update user
   */
  async updateUser(userId: string, updates: Partial<CreateUserRequest>): Promise<User> {
    try {
      const response = await this.httpClient.patch(`/users/${userId}`, updates);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to update user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Update current user profile
   */
  async updateCurrentUser(updates: { 
    first_name?: string; 
    last_name?: string; 
    email?: string 
  }): Promise<User> {
    try {
      const response = await this.httpClient.patch('/users/me', updates);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to update current user profile', 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Delete user
   */
  async deleteUser(userId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/users/${userId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to delete user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Activate user
   */
  async activateUser(userId: string): Promise<User> {
    try {
      const response = await this.httpClient.post(`/users/${userId}/activate`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to activate user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Deactivate user
   */
  async deactivateUser(userId: string): Promise<User> {
    try {
      const response = await this.httpClient.post(`/users/${userId}/deactivate`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to deactivate user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Change user password
   */
  async changePassword(userId: string, currentPassword: string, newPassword: string): Promise<void> {
    try {
      await this.httpClient.post(`/users/${userId}/password`, {
        current_password: currentPassword,
        new_password: newPassword
      });
    } catch (error) {
      throw error instanceof PynomalyError ? error : new AuthenticationError('Failed to change password');
    }
  }

  /**
   * Change current user password
   */
  async changeCurrentPassword(currentPassword: string, newPassword: string): Promise<void> {
    try {
      await this.httpClient.post('/users/me/password', {
        current_password: currentPassword,
        new_password: newPassword
      });
    } catch (error) {
      throw error instanceof PynomalyError ? error : new AuthenticationError('Failed to change current password');
    }
  }

  /**
   * Reset user password
   */
  async resetPassword(userId: string): Promise<{ temporary_password: string }> {
    try {
      const response = await this.httpClient.post(`/users/${userId}/password/reset`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to reset password for user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Send password reset email
   */
  async sendPasswordResetEmail(email: string): Promise<void> {
    try {
      await this.httpClient.post('/users/password/reset-email', { email });
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to send password reset email', 'USER_MANAGEMENT_ERROR');
    }
  }

  // Role Management

  /**
   * Get user roles
   */
  async getUserRoles(userId: string): Promise<UserRole[]> {
    try {
      const response = await this.httpClient.get(`/users/${userId}/roles`);
      return response.data.roles || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get roles for user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Add role to user
   */
  async addUserRole(userId: string, role: UserRole): Promise<User> {
    try {
      const response = await this.httpClient.post(`/users/${userId}/roles`, { role });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new AuthorizationError(`Failed to add role ${role} to user ${userId}`);
    }
  }

  /**
   * Remove role from user
   */
  async removeUserRole(userId: string, role: UserRole): Promise<User> {
    try {
      const response = await this.httpClient.delete(`/users/${userId}/roles/${role}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new AuthorizationError(`Failed to remove role ${role} from user ${userId}`);
    }
  }

  /**
   * Check if user has role
   */
  async hasRole(userId: string, role: UserRole): Promise<boolean> {
    try {
      const response = await this.httpClient.get(`/users/${userId}/roles/${role}`);
      return response.data.has_role;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to check role ${role} for user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Get all available roles
   */
  async getAvailableRoles(): Promise<Array<{ role: UserRole; description: string; permissions: string[] }>> {
    try {
      const response = await this.httpClient.get('/roles');
      return response.data.roles || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to get available roles', 'USER_MANAGEMENT_ERROR');
    }
  }

  // Tenant Management

  /**
   * List all tenants
   */
  async listTenants(options: ListOptions = {}): Promise<PaginatedResponse<Tenant>> {
    try {
      const response = await this.httpClient.get('/tenants', { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to list tenants', 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Get tenant by ID
   */
  async getTenant(tenantId: string): Promise<Tenant> {
    try {
      const response = await this.httpClient.get(`/tenants/${tenantId}`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Get current tenant
   */
  async getCurrentTenant(): Promise<Tenant> {
    try {
      const response = await this.httpClient.get('/tenants/current');
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to get current tenant', 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Create new tenant
   */
  async createTenant(request: {
    name: string;
    description: string;
    subscription_tier: string;
    max_users?: number;
    max_detectors?: number;
    max_datasets?: number;
  }): Promise<Tenant> {
    try {
      const response = await this.httpClient.post('/tenants', request);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to create tenant', 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Update tenant
   */
  async updateTenant(tenantId: string, updates: {
    name?: string;
    description?: string;
    subscription_tier?: string;
    max_users?: number;
    max_detectors?: number;
    max_datasets?: number;
  }): Promise<Tenant> {
    try {
      const response = await this.httpClient.patch(`/tenants/${tenantId}`, updates);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to update tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Delete tenant
   */
  async deleteTenant(tenantId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/tenants/${tenantId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to delete tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Activate tenant
   */
  async activateTenant(tenantId: string): Promise<Tenant> {
    try {
      const response = await this.httpClient.post(`/tenants/${tenantId}/activate`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to activate tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Deactivate tenant
   */
  async deactivateTenant(tenantId: string): Promise<Tenant> {
    try {
      const response = await this.httpClient.post(`/tenants/${tenantId}/deactivate`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to deactivate tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Get tenant users
   */
  async getTenantUsers(tenantId: string, options: ListOptions = {}): Promise<PaginatedResponse<User>> {
    try {
      const response = await this.httpClient.get(`/tenants/${tenantId}/users`, { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get users for tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Add user to tenant
   */
  async addUserToTenant(tenantId: string, userId: string, roles: UserRole[] = []): Promise<void> {
    try {
      await this.httpClient.post(`/tenants/${tenantId}/users`, { user_id: userId, roles });
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to add user ${userId} to tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Remove user from tenant
   */
  async removeUserFromTenant(tenantId: string, userId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/tenants/${tenantId}/users/${userId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to remove user ${userId} from tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Get tenant usage statistics
   */
  async getTenantUsage(tenantId: string): Promise<{
    users: { current: number; max: number };
    detectors: { current: number; max: number };
    datasets: { current: number; max: number };
    storage_used_gb: number;
    api_calls_this_month: number;
    last_activity: string;
  }> {
    try {
      const response = await this.httpClient.get(`/tenants/${tenantId}/usage`);
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get usage for tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Get tenant activity log
   */
  async getTenantActivity(tenantId: string, options: ListOptions = {}): Promise<PaginatedResponse<{
    id: string;
    timestamp: string;
    user_id: string;
    action: string;
    resource_type: string;
    resource_id: string;
    details: Record<string, any>;
  }>> {
    try {
      const response = await this.httpClient.get(`/tenants/${tenantId}/activity`, { params: options });
      return response.data;
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get activity for tenant ${tenantId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  // Session Management

  /**
   * Get active sessions for user
   */
  async getUserSessions(userId: string): Promise<Array<{
    id: string;
    created_at: string;
    last_activity: string;
    ip_address: string;
    user_agent: string;
    is_current: boolean;
  }>> {
    try {
      const response = await this.httpClient.get(`/users/${userId}/sessions`);
      return response.data.sessions || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to get sessions for user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Revoke user session
   */
  async revokeUserSession(userId: string, sessionId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/users/${userId}/sessions/${sessionId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to revoke session ${sessionId} for user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Revoke all user sessions
   */
  async revokeAllUserSessions(userId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/users/${userId}/sessions`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to revoke all sessions for user ${userId}`, 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Get current user sessions
   */
  async getCurrentUserSessions(): Promise<Array<{
    id: string;
    created_at: string;
    last_activity: string;
    ip_address: string;
    user_agent: string;
    is_current: boolean;
  }>> {
    try {
      const response = await this.httpClient.get('/users/me/sessions');
      return response.data.sessions || [];
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError('Failed to get current user sessions', 'USER_MANAGEMENT_ERROR');
    }
  }

  /**
   * Revoke current user session
   */
  async revokeCurrentUserSession(sessionId: string): Promise<void> {
    try {
      await this.httpClient.delete(`/users/me/sessions/${sessionId}`);
    } catch (error) {
      throw error instanceof PynomalyError ? error : new PynomalyError(`Failed to revoke current user session ${sessionId}`, 'USER_MANAGEMENT_ERROR');
    }
  }
}