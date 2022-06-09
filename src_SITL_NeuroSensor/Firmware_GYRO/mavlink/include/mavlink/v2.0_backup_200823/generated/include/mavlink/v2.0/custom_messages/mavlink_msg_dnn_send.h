#pragma once
// MESSAGE DNN_SEND PACKING

#define MAVLINK_MSG_ID_DNN_SEND 12010


typedef struct __mavlink_dnn_send_t {
 uint64_t time_usec; /*< [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.*/
 int32_t attack_trigger; /*<  attack trigger*/
 float compromised_signal; /*<  input gyro signal*/
} mavlink_dnn_send_t;

#define MAVLINK_MSG_ID_DNN_SEND_LEN 16
#define MAVLINK_MSG_ID_DNN_SEND_MIN_LEN 16
#define MAVLINK_MSG_ID_12010_LEN 16
#define MAVLINK_MSG_ID_12010_MIN_LEN 16

#define MAVLINK_MSG_ID_DNN_SEND_CRC 62
#define MAVLINK_MSG_ID_12010_CRC 62



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_DNN_SEND { \
    12010, \
    "DNN_SEND", \
    3, \
    {  { "time_usec", NULL, MAVLINK_TYPE_UINT64_T, 0, 0, offsetof(mavlink_dnn_send_t, time_usec) }, \
         { "attack_trigger", NULL, MAVLINK_TYPE_INT32_T, 0, 8, offsetof(mavlink_dnn_send_t, attack_trigger) }, \
         { "compromised_signal", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_dnn_send_t, compromised_signal) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_DNN_SEND { \
    "DNN_SEND", \
    3, \
    {  { "time_usec", NULL, MAVLINK_TYPE_UINT64_T, 0, 0, offsetof(mavlink_dnn_send_t, time_usec) }, \
         { "attack_trigger", NULL, MAVLINK_TYPE_INT32_T, 0, 8, offsetof(mavlink_dnn_send_t, attack_trigger) }, \
         { "compromised_signal", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_dnn_send_t, compromised_signal) }, \
         } \
}
#endif

/**
 * @brief Pack a dnn_send message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param time_usec [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.
 * @param attack_trigger  attack trigger
 * @param compromised_signal  input gyro signal
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_dnn_send_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               uint64_t time_usec, int32_t attack_trigger, float compromised_signal)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_DNN_SEND_LEN];
    _mav_put_uint64_t(buf, 0, time_usec);
    _mav_put_int32_t(buf, 8, attack_trigger);
    _mav_put_float(buf, 12, compromised_signal);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_DNN_SEND_LEN);
#else
    mavlink_dnn_send_t packet;
    packet.time_usec = time_usec;
    packet.attack_trigger = attack_trigger;
    packet.compromised_signal = compromised_signal;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_DNN_SEND_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_DNN_SEND;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_DNN_SEND_MIN_LEN, MAVLINK_MSG_ID_DNN_SEND_LEN, MAVLINK_MSG_ID_DNN_SEND_CRC);
}

/**
 * @brief Pack a dnn_send message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param time_usec [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.
 * @param attack_trigger  attack trigger
 * @param compromised_signal  input gyro signal
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_dnn_send_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   uint64_t time_usec,int32_t attack_trigger,float compromised_signal)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_DNN_SEND_LEN];
    _mav_put_uint64_t(buf, 0, time_usec);
    _mav_put_int32_t(buf, 8, attack_trigger);
    _mav_put_float(buf, 12, compromised_signal);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_DNN_SEND_LEN);
#else
    mavlink_dnn_send_t packet;
    packet.time_usec = time_usec;
    packet.attack_trigger = attack_trigger;
    packet.compromised_signal = compromised_signal;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_DNN_SEND_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_DNN_SEND;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_DNN_SEND_MIN_LEN, MAVLINK_MSG_ID_DNN_SEND_LEN, MAVLINK_MSG_ID_DNN_SEND_CRC);
}

/**
 * @brief Encode a dnn_send struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param dnn_send C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_dnn_send_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_dnn_send_t* dnn_send)
{
    return mavlink_msg_dnn_send_pack(system_id, component_id, msg, dnn_send->time_usec, dnn_send->attack_trigger, dnn_send->compromised_signal);
}

/**
 * @brief Encode a dnn_send struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param dnn_send C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_dnn_send_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_dnn_send_t* dnn_send)
{
    return mavlink_msg_dnn_send_pack_chan(system_id, component_id, chan, msg, dnn_send->time_usec, dnn_send->attack_trigger, dnn_send->compromised_signal);
}

/**
 * @brief Send a dnn_send message
 * @param chan MAVLink channel to send the message
 *
 * @param time_usec [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.
 * @param attack_trigger  attack trigger
 * @param compromised_signal  input gyro signal
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_dnn_send_send(mavlink_channel_t chan, uint64_t time_usec, int32_t attack_trigger, float compromised_signal)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_DNN_SEND_LEN];
    _mav_put_uint64_t(buf, 0, time_usec);
    _mav_put_int32_t(buf, 8, attack_trigger);
    _mav_put_float(buf, 12, compromised_signal);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_SEND, buf, MAVLINK_MSG_ID_DNN_SEND_MIN_LEN, MAVLINK_MSG_ID_DNN_SEND_LEN, MAVLINK_MSG_ID_DNN_SEND_CRC);
#else
    mavlink_dnn_send_t packet;
    packet.time_usec = time_usec;
    packet.attack_trigger = attack_trigger;
    packet.compromised_signal = compromised_signal;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_SEND, (const char *)&packet, MAVLINK_MSG_ID_DNN_SEND_MIN_LEN, MAVLINK_MSG_ID_DNN_SEND_LEN, MAVLINK_MSG_ID_DNN_SEND_CRC);
#endif
}

/**
 * @brief Send a dnn_send message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_dnn_send_send_struct(mavlink_channel_t chan, const mavlink_dnn_send_t* dnn_send)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_dnn_send_send(chan, dnn_send->time_usec, dnn_send->attack_trigger, dnn_send->compromised_signal);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_SEND, (const char *)dnn_send, MAVLINK_MSG_ID_DNN_SEND_MIN_LEN, MAVLINK_MSG_ID_DNN_SEND_LEN, MAVLINK_MSG_ID_DNN_SEND_CRC);
#endif
}

#if MAVLINK_MSG_ID_DNN_SEND_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_dnn_send_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  uint64_t time_usec, int32_t attack_trigger, float compromised_signal)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_uint64_t(buf, 0, time_usec);
    _mav_put_int32_t(buf, 8, attack_trigger);
    _mav_put_float(buf, 12, compromised_signal);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_SEND, buf, MAVLINK_MSG_ID_DNN_SEND_MIN_LEN, MAVLINK_MSG_ID_DNN_SEND_LEN, MAVLINK_MSG_ID_DNN_SEND_CRC);
#else
    mavlink_dnn_send_t *packet = (mavlink_dnn_send_t *)msgbuf;
    packet->time_usec = time_usec;
    packet->attack_trigger = attack_trigger;
    packet->compromised_signal = compromised_signal;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_SEND, (const char *)packet, MAVLINK_MSG_ID_DNN_SEND_MIN_LEN, MAVLINK_MSG_ID_DNN_SEND_LEN, MAVLINK_MSG_ID_DNN_SEND_CRC);
#endif
}
#endif

#endif

// MESSAGE DNN_SEND UNPACKING


/**
 * @brief Get field time_usec from dnn_send message
 *
 * @return [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.
 */
static inline uint64_t mavlink_msg_dnn_send_get_time_usec(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint64_t(msg,  0);
}

/**
 * @brief Get field attack_trigger from dnn_send message
 *
 * @return  attack trigger
 */
static inline int32_t mavlink_msg_dnn_send_get_attack_trigger(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int32_t(msg,  8);
}

/**
 * @brief Get field compromised_signal from dnn_send message
 *
 * @return  input gyro signal
 */
static inline float mavlink_msg_dnn_send_get_compromised_signal(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  12);
}

/**
 * @brief Decode a dnn_send message into a struct
 *
 * @param msg The message to decode
 * @param dnn_send C-struct to decode the message contents into
 */
static inline void mavlink_msg_dnn_send_decode(const mavlink_message_t* msg, mavlink_dnn_send_t* dnn_send)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    dnn_send->time_usec = mavlink_msg_dnn_send_get_time_usec(msg);
    dnn_send->attack_trigger = mavlink_msg_dnn_send_get_attack_trigger(msg);
    dnn_send->compromised_signal = mavlink_msg_dnn_send_get_compromised_signal(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_DNN_SEND_LEN? msg->len : MAVLINK_MSG_ID_DNN_SEND_LEN;
        memset(dnn_send, 0, MAVLINK_MSG_ID_DNN_SEND_LEN);
    memcpy(dnn_send, _MAV_PAYLOAD(msg), len);
#endif
}
