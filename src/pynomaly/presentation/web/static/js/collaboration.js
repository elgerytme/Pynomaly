/**
 * Collaboration JavaScript
 * Handles real-time collaboration features
 */

// Global collaboration manager
window.CollaborationManager = {
    socket: null,
    currentUser: null,
    activeUsers: new Map(),
    chatMessages: [],
    cursors: new Map(),
    selections: new Map(),
    activities: [],
    
    init() {
        this.initializeUser();
        this.connectWebSocket();
        this.setupEventListeners();
        this.initChatSystem();
        this.initPresenceSystem();
        this.startActivityTracking();
    },
    
    initializeUser() {
        // Get current user info (simulated)
        this.currentUser = {
            id: 'user_' + Math.random().toString(36).substr(2, 9),
            name: 'User ' + Math.floor(Math.random() * 1000),
            avatar: this.generateAvatar(),
            color: this.generateUserColor(),
            status: 'online',
            joinedAt: new Date()
        };
        
        // Update UI
        this.updateUserInfo();
    },
    
    connectWebSocket() {
        // Simulate WebSocket connection
        console.log('Connecting to collaboration server...');
        
        // In a real implementation, this would connect to a WebSocket server
        this.socket = {
            send: (data) => {
                console.log('Sending:', data);
                // Simulate echo back for testing
                setTimeout(() => {
                    this.handleMessage(JSON.parse(data));
                }, 100);
            },
            close: () => {
                console.log('WebSocket closed');
            }
        };
        
        // Simulate successful connection
        setTimeout(() => {
            this.onConnected();
        }, 500);
    },
    
    onConnected() {
        console.log('Connected to collaboration server');
        
        // Join room
        this.sendMessage({
            type: 'join_room',
            user: this.currentUser,
            room: 'pynomaly_workspace'
        });
        
        // Simulate other users
        this.simulateOtherUsers();
    },
    
    simulateOtherUsers() {
        const users = [
            { name: 'Alice Johnson', status: 'online' },
            { name: 'Bob Smith', status: 'away' },
            { name: 'Carol Davis', status: 'online' },
            { name: 'David Wilson', status: 'offline' }
        ];
        
        users.forEach((user, index) => {
            setTimeout(() => {
                const simulatedUser = {
                    id: 'user_' + Math.random().toString(36).substr(2, 9),
                    name: user.name,
                    avatar: this.generateAvatar(),
                    color: this.generateUserColor(),
                    status: user.status,
                    joinedAt: new Date(Date.now() - Math.random() * 3600000)
                };
                
                this.activeUsers.set(simulatedUser.id, simulatedUser);
                this.updateUsersList();
                
                // Simulate activity
                if (user.status === 'online') {
                    this.simulateUserActivity(simulatedUser);
                }
            }, index * 500);
        });
    },
    
    simulateUserActivity(user) {
        const activities = [
            'is viewing anomaly detection results',
            'is training a new detector',
            'is analyzing feature correlations',
            'is configuring ensemble parameters',
            'is reviewing performance metrics'
        ];
        
        setInterval(() => {
            if (Math.random() > 0.7) { // 30% chance of activity
                const activity = activities[Math.floor(Math.random() * activities.length)];
                this.addActivity({
                    user: user,
                    action: activity,
                    timestamp: new Date(),
                    type: 'action'
                });
            }
        }, 5000 + Math.random() * 10000);
    },
    
    setupEventListeners() {
        // Chat form submission
        const chatForm = document.getElementById('chat-form');
        if (chatForm) {
            chatForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.sendChatMessage();
            });
        }
        
        // Mouse tracking for cursors
        document.addEventListener('mousemove', (e) => {
            this.trackCursor(e);
        });
        
        // Selection tracking
        document.addEventListener('mouseup', () => {
            this.trackSelection();
        });
        
        // Typing indicators
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            let typingTimeout;
            chatInput.addEventListener('input', () => {
                this.showTypingIndicator();
                clearTimeout(typingTimeout);
                typingTimeout = setTimeout(() => {
                    this.hideTypingIndicator();
                }, 1000);
            });
        }
        
        // Status change
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action="change-status"]')) {
                const status = e.target.dataset.status;
                this.changeUserStatus(status);
            }
        });
        
        // User actions
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action="mention-user"]')) {
                const userId = e.target.dataset.userId;
                this.mentionUser(userId);
            }
            
            if (e.target.matches('[data-action="start-video-call"]')) {
                this.startVideoCall();
            }
            
            if (e.target.matches('[data-action="share-screen"]')) {
                this.shareScreen();
            }
        });
    },
    
    initChatSystem() {
        // Load chat history (simulated)
        this.loadChatHistory();
        
        // Setup chat UI
        this.updateChatDisplay();
    },
    
    loadChatHistory() {
        // Simulate loading previous messages
        const sampleMessages = [
            {
                id: 'msg_1',
                user: { name: 'Alice Johnson', color: '#3B82F6' },
                content: 'Has anyone tested the new ensemble detector yet?',
                timestamp: new Date(Date.now() - 1800000),
                type: 'message'
            },
            {
                id: 'msg_2',
                user: { name: 'Bob Smith', color: '#10B981' },
                content: 'Yes, it\'s showing great results on the financial dataset!',
                timestamp: new Date(Date.now() - 1700000),
                type: 'message'
            },
            {
                id: 'msg_3',
                user: { name: 'System', color: '#6B7280' },
                content: 'Carol Davis joined the workspace',
                timestamp: new Date(Date.now() - 1600000),
                type: 'system'
            }
        ];
        
        this.chatMessages = sampleMessages;
    },
    
    sendChatMessage() {
        const input = document.getElementById('chat-input');
        if (!input || !input.value.trim()) return;
        
        const message = {
            id: 'msg_' + Date.now(),
            user: this.currentUser,
            content: input.value.trim(),
            timestamp: new Date(),
            type: 'message'
        };
        
        this.chatMessages.push(message);
        this.updateChatDisplay();
        
        // Send to server
        this.sendMessage({
            type: 'chat_message',
            message: message
        });
        
        input.value = '';
    },
    
    updateChatDisplay() {
        const chatContainer = document.getElementById('chat-messages');
        if (!chatContainer) return;
        
        chatContainer.innerHTML = '';
        
        this.chatMessages.forEach(message => {
            const messageElement = this.createMessageElement(message);
            chatContainer.appendChild(messageElement);
        });
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    },
    
    createMessageElement(message) {
        const div = document.createElement('div');
        div.className = 'chat-message mb-3';
        
        if (message.type === 'system') {
            div.innerHTML = `
                <div class="text-center text-xs text-gray-500 py-1">
                    ${message.content}
                </div>
            `;
        } else {
            const isOwnMessage = message.user.id === this.currentUser.id;
            div.innerHTML = `
                <div class="flex items-start space-x-2 ${isOwnMessage ? 'flex-row-reverse space-x-reverse' : ''}">
                    <div class="user-avatar" style="background-color: ${message.user.color}">
                        ${message.user.name.charAt(0)}
                    </div>
                    <div class="flex-1 ${isOwnMessage ? 'text-right' : ''}">
                        <div class="flex items-center space-x-2 ${isOwnMessage ? 'justify-end' : ''}">
                            <span class="text-sm font-medium text-gray-900">${message.user.name}</span>
                            <span class="text-xs text-gray-500">${this.formatTime(message.timestamp)}</span>
                        </div>
                        <div class="mt-1 ${isOwnMessage ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-900'} rounded-lg px-3 py-2 inline-block max-w-xs">
                            ${this.formatMessageContent(message.content)}
                        </div>
                    </div>
                </div>
            `;
        }
        
        return div;
    },
    
    formatMessageContent(content) {
        // Handle mentions, links, etc.
        return content
            .replace(/@(\w+)/g, '<span class="mention">@$1</span>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" class="text-blue-600 underline">$1</a>');
    },
    
    formatTime(date) {
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) {
            return 'just now';
        } else if (diff < 3600000) {
            return Math.floor(diff / 60000) + 'm ago';
        } else if (diff < 86400000) {
            return Math.floor(diff / 3600000) + 'h ago';
        } else {
            return date.toLocaleDateString();
        }
    },
    
    initPresenceSystem() {
        this.updateUsersList();
        this.startPresenceHeartbeat();
    },
    
    updateUsersList() {
        const container = document.getElementById('active-users');
        if (!container) return;
        
        container.innerHTML = '';
        
        // Add current user first
        const currentUserElement = this.createUserElement(this.currentUser, true);
        container.appendChild(currentUserElement);
        
        // Add other users
        for (const [userId, user] of this.activeUsers) {
            const userElement = this.createUserElement(user, false);
            container.appendChild(userElement);
        }
    },
    
    createUserElement(user, isSelf) {
        const div = document.createElement('div');
        div.className = 'flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50';
        div.innerHTML = `
            <div class="relative">
                <div class="user-avatar user-status-${user.status}" style="background-color: ${user.color}">
                    ${user.name.charAt(0)}
                </div>
                ${user.status === 'online' ? '<div class="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 border-2 border-white rounded-full"></div>' : ''}
            </div>
            <div class="flex-1 min-w-0">
                <p class="text-sm font-medium text-gray-900 truncate">
                    ${user.name} ${isSelf ? '(You)' : ''}
                </p>
                <p class="text-xs text-gray-500 capitalize">${user.status}</p>
            </div>
            <div class="flex space-x-1">
                <button class="text-xs bg-blue-100 text-blue-600 px-2 py-1 rounded hover:bg-blue-200"
                        data-action="mention-user" data-user-id="${user.id}">
                    @
                </button>
                ${user.status === 'online' && !isSelf ? `
                    <button class="text-xs bg-green-100 text-green-600 px-2 py-1 rounded hover:bg-green-200"
                            data-action="start-video-call" data-user-id="${user.id}">
                        📹
                    </button>
                ` : ''}
            </div>
        `;
        
        return div;
    },
    
    startPresenceHeartbeat() {
        setInterval(() => {
            this.sendMessage({
                type: 'heartbeat',
                user: this.currentUser
            });
        }, 30000); // Every 30 seconds
    },
    
    trackCursor(e) {
        // Update own cursor position
        const cursor = {
            x: e.clientX,
            y: e.clientY,
            user: this.currentUser,
            timestamp: Date.now()
        };
        
        // Send to other users
        this.sendMessage({
            type: 'cursor_move',
            cursor: cursor
        });
    },
    
    updateRemoteCursor(cursorData) {
        const cursorsContainer = document.getElementById('remote-cursors');
        if (!cursorsContainer) return;
        
        let cursorElement = document.getElementById(`cursor-${cursorData.user.id}`);
        
        if (!cursorElement) {
            cursorElement = document.createElement('div');
            cursorElement.id = `cursor-${cursorData.user.id}`;
            cursorElement.className = 'live-cursor';
            cursorElement.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L13.09 8.26L17 12L13.09 15.74L12 22L10.91 15.74L7 12L10.91 8.26L12 2Z" 
                          fill="${cursorData.user.color}"/>
                </svg>
                <div class="absolute top-5 left-3 bg-black text-white text-xs px-2 py-1 rounded whitespace-nowrap">
                    ${cursorData.user.name}
                </div>
            `;
            cursorsContainer.appendChild(cursorElement);
        }
        
        cursorElement.style.left = cursorData.x + 'px';
        cursorElement.style.top = cursorData.y + 'px';
        
        // Store cursor data
        this.cursors.set(cursorData.user.id, cursorData);
    },
    
    trackSelection() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            
            const selectionData = {
                user: this.currentUser,
                x: rect.left,
                y: rect.top,
                width: rect.width,
                height: rect.height,
                text: selection.toString(),
                timestamp: Date.now()
            };
            
            this.sendMessage({
                type: 'selection_change',
                selection: selectionData
            });
        }
    },
    
    updateRemoteSelection(selectionData) {
        const selectionsContainer = document.getElementById('remote-selections');
        if (!selectionsContainer) return;
        
        let selectionElement = document.getElementById(`selection-${selectionData.user.id}`);
        
        if (!selectionElement) {
            selectionElement = document.createElement('div');
            selectionElement.id = `selection-${selectionData.user.id}`;
            selectionElement.className = 'live-selection';
            selectionsContainer.appendChild(selectionElement);
        }
        
        selectionElement.style.left = selectionData.x + 'px';
        selectionElement.style.top = selectionData.y + 'px';
        selectionElement.style.width = selectionData.width + 'px';
        selectionElement.style.height = selectionData.height + 'px';
        selectionElement.style.borderColor = selectionData.user.color;
        
        // Store selection data
        this.selections.set(selectionData.user.id, selectionData);
    },
    
    showTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.style.display = 'flex';
        }
        
        this.sendMessage({
            type: 'typing_start',
            user: this.currentUser
        });
    },
    
    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
        
        this.sendMessage({
            type: 'typing_stop',
            user: this.currentUser
        });
    },
    
    startActivityTracking() {
        // Track page interactions
        this.trackPageViews();
        this.trackDetectorActions();
        this.trackDatasetActions();
    },
    
    trackPageViews() {
        // Track when user navigates to different pages
        const currentPage = window.location.pathname;
        this.addActivity({
            user: this.currentUser,
            action: `is viewing ${this.getPageName(currentPage)}`,
            timestamp: new Date(),
            type: 'navigation'
        });
    },
    
    trackDetectorActions() {
        // Listen for detector-related actions
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action="train-detector"]')) {
                this.addActivity({
                    user: this.currentUser,
                    action: 'started training a detector',
                    timestamp: new Date(),
                    type: 'detector_action'
                });
            }
            
            if (e.target.matches('[data-action="detect-anomalies"]')) {
                this.addActivity({
                    user: this.currentUser,
                    action: 'started anomaly detection',
                    timestamp: new Date(),
                    type: 'detection_action'
                });
            }
        });
    },
    
    trackDatasetActions() {
        // Listen for dataset-related actions
        document.addEventListener('change', (e) => {
            if (e.target.matches('[name="dataset_id"]')) {
                this.addActivity({
                    user: this.currentUser,
                    action: 'selected a dataset',
                    timestamp: new Date(),
                    type: 'dataset_action'
                });
            }
        });
    },
    
    addActivity(activity) {
        this.activities.unshift(activity);
        
        // Keep only last 50 activities
        if (this.activities.length > 50) {
            this.activities = this.activities.slice(0, 50);
        }
        
        this.updateActivityFeed();
        
        // Send to other users
        this.sendMessage({
            type: 'activity',
            activity: activity
        });
    },
    
    updateActivityFeed() {
        const container = document.getElementById('activity-feed');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.activities.slice(0, 20).forEach(activity => {
            const activityElement = this.createActivityElement(activity);
            container.appendChild(activityElement);
        });
    },
    
    createActivityElement(activity) {
        const div = document.createElement('div');
        div.className = 'activity-item py-2 border-b border-gray-100 last:border-b-0';
        div.innerHTML = `
            <div class="flex items-center space-x-2">
                <div class="user-avatar" style="background-color: ${activity.user.color}; width: 24px; height: 24px; font-size: 10px;">
                    ${activity.user.name.charAt(0)}
                </div>
                <div class="flex-1 min-w-0">
                    <p class="text-sm text-gray-900">
                        <span class="font-medium">${activity.user.name}</span>
                        <span class="text-gray-600">${activity.action}</span>
                    </p>
                    <p class="text-xs text-gray-500">${this.formatTime(activity.timestamp)}</p>
                </div>
            </div>
        `;
        
        return div;
    },
    
    changeUserStatus(status) {
        this.currentUser.status = status;
        this.updateUserInfo();
        this.updateUsersList();
        
        this.sendMessage({
            type: 'status_change',
            user: this.currentUser,
            status: status
        });
    },
    
    mentionUser(userId) {
        const chatInput = document.getElementById('chat-input');
        if (!chatInput) return;
        
        const user = this.activeUsers.get(userId) || this.currentUser;
        const mention = `@${user.name} `;
        
        chatInput.value += mention;
        chatInput.focus();
    },
    
    startVideoCall() {
        // Simulate starting a video call
        this.addActivity({
            user: this.currentUser,
            action: 'started a video call',
            timestamp: new Date(),
            type: 'communication'
        });
        
        alert('Video call feature would be implemented with WebRTC');
    },
    
    shareScreen() {
        // Simulate screen sharing
        this.addActivity({
            user: this.currentUser,
            action: 'started screen sharing',
            timestamp: new Date(),
            type: 'communication'
        });
        
        alert('Screen sharing feature would be implemented with WebRTC');
    },
    
    sendMessage(data) {
        if (this.socket && this.socket.send) {
            this.socket.send(JSON.stringify(data));
        }
    },
    
    handleMessage(data) {
        switch (data.type) {
            case 'chat_message':
                if (data.message.user.id !== this.currentUser.id) {
                    this.chatMessages.push(data.message);
                    this.updateChatDisplay();
                }
                break;
                
            case 'cursor_move':
                if (data.cursor.user.id !== this.currentUser.id) {
                    this.updateRemoteCursor(data.cursor);
                }
                break;
                
            case 'selection_change':
                if (data.selection.user.id !== this.currentUser.id) {
                    this.updateRemoteSelection(data.selection);
                }
                break;
                
            case 'activity':
                if (data.activity.user.id !== this.currentUser.id) {
                    this.activities.unshift(data.activity);
                    this.updateActivityFeed();
                }
                break;
                
            case 'user_joined':
                this.activeUsers.set(data.user.id, data.user);
                this.updateUsersList();
                this.addSystemMessage(`${data.user.name} joined the workspace`);
                break;
                
            case 'user_left':
                this.activeUsers.delete(data.user.id);
                this.updateUsersList();
                this.addSystemMessage(`${data.user.name} left the workspace`);
                break;
                
            case 'status_change':
                if (this.activeUsers.has(data.user.id)) {
                    this.activeUsers.get(data.user.id).status = data.status;
                    this.updateUsersList();
                }
                break;
        }
    },
    
    addSystemMessage(content) {
        const message = {
            id: 'sys_' + Date.now(),
            user: { name: 'System', color: '#6B7280' },
            content: content,
            timestamp: new Date(),
            type: 'system'
        };
        
        this.chatMessages.push(message);
        this.updateChatDisplay();
    },
    
    updateUserInfo() {
        const userInfo = document.getElementById('current-user-info');
        if (userInfo) {
            userInfo.innerHTML = `
                <div class="flex items-center space-x-3">
                    <div class="user-avatar user-status-${this.currentUser.status}" style="background-color: ${this.currentUser.color}">
                        ${this.currentUser.name.charAt(0)}
                    </div>
                    <div>
                        <p class="font-medium text-gray-900">${this.currentUser.name}</p>
                        <p class="text-sm text-gray-500 capitalize">${this.currentUser.status}</p>
                    </div>
                </div>
            `;
        }
    },
    
    generateAvatar() {
        // Generate a simple avatar based on user name
        return Math.random().toString(36).substr(2, 2).toUpperCase();
    },
    
    generateUserColor() {
        const colors = [
            '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
            '#EC4899', '#14B8A6', '#F97316', '#84CC16', '#6366F1'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    },
    
    getPageName(path) {
        const pageNames = {
            '/': 'Dashboard',
            '/detectors': 'Detectors',
            '/datasets': 'Datasets',
            '/detection': 'Detection',
            '/ensemble': 'Ensemble',
            '/automl': 'AutoML',
            '/explainability': 'Explainability',
            '/monitoring': 'Monitoring',
            '/visualizations': 'Visualizations',
            '/workflows': 'Workflows',
            '/collaboration': 'Collaboration',
            '/advanced-visualizations': 'Advanced Visualizations'
        };
        
        return pageNames[path] || 'Unknown Page';
    },
    
    // Cleanup
    destroy() {
        if (this.socket && this.socket.close) {
            this.socket.close();
        }
        
        // Clear intervals and timeouts
        // In a real implementation, you'd track these and clear them
    }
};

// Alpine.js data for collaboration
function collaborationHub() {
    return {
        activeTab: 'chat',
        unreadMessages: 0,
        
        switchTab(tab) {
            this.activeTab = tab;
            if (tab === 'chat') {
                this.unreadMessages = 0;
            }
        },
        
        get hasUnreadMessages() {
            return this.unreadMessages > 0;
        }
    };
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (typeof CollaborationManager !== 'undefined') {
        CollaborationManager.init();
    }
});
