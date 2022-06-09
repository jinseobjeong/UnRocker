#pragma once
// MESSAGE DNN_RECV PACKING

#define MAVLINK_MSG_ID_DNN_RECV 12011


typedef struct __mavlink_dnn_recv_t {
 float recovered_signal; /*<  output gyro signal*/
} mavlink_dnn_recv_t;

#define MAVLINK_MSG_ID_DNN_RECV_LEN 4
#define MAVLINK_MSG_ID_DNN_RECV_MIN_LEN 4
#define MAVLINK_MSG_ID_12011_LEN 4
#define MAVLINK_MSG_ID_12011_MIN_LEN 4

#define MAVLINK_MSG_ID_DNN_RECV_CRC 143
#define MAVLINK_MSG_ID_12011_CRC 143



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_DNN_RECV { \
    12011, \
    "DNN_RECV", \
    1, \
    {  { "recovered_signal", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_dnn_recv_t, recovered_signal) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_DNN_RECV { \
    "DNN_RECV", \
    1, \
    {  { "recovered_signal", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_dnn_recv_t, recovered_signal) }, \
         } \
}
#endif

/**
 * @brief Pack a dnn_recv message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param recovered_signal  output gyro signal
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_dnn_recv_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               float recovered_signal)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_DNN_RECV_LEN];
    _mav_put_float(buf, 0, recovered_signal);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_DNN_RECV_LEN);
#else
    mavlink_dnn_recv_t packet;
    packet.recovered_signal = recovered_signal;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_DNN_RECV_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_DNN_RECV;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_DNN_RECV_MIN_LEN, MAVLINK_MSG_ID_DNN_RECV_LEN, MAVLINK_MSG_ID_DNN_RECV_CRC);
}

/**
 * @brief Pack a dnn_recv message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param recovered_signal  output gyro signal
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_dnn_recv_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   float recovered_signal)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_DNN_RECV_LEN];
    _mav_put_float(buf, 0, recovered_signal);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_DNN_RECV_LEN);
#else
    mavlink_dnn_recv_t packet;
    packet.recovered_signal = recovered_signal;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_DNN_RECV_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_DNN_RECV;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_DNN_RECV_MIN_LEN, MAVLINK_MSG_ID_DNN_RECV_LEN, MAVLINK_MSG_ID_DNN_RECV_CRC);
}

/**
 * @brief Encode a dnn_recv struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param dnn_recv C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_dnn_recv_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_dnn_recv_t* dnn_recv)
{
    return mavlink_msg_dnn_recv_pack(system_id, component_id, msg, dnn_recv->recovered_signal);
}

/**
 * @brief Encode a dnn_recv struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param dnn_recv C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_dnn_recv_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_dnn_recv_t* dnn_recv)
{
    return mavlink_msg_dnn_recv_pack_chan(system_id, component_id, chan, msg, dnn_recv->recovered_signal);
}

/**
 * @brief Send a dnn_recv message
 * @param chan MAVLink channel to send the message
 *
 * @param recovered_signal  output gyro signal
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_dnn_recv_send(mavlink_channel_t chan, float recovered_signal)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_DNN_RECV_LEN];
    _mav_put_float(buf, 0, recovered_signal);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_RECV, buf, MAVLINK_MSG_ID_DNN_RECV_MIN_LEN, MAVLINK_MSG_ID_DNN_RECV_LEN, MAVLINK_MSG_ID_DNN_RECV_CRC);
#else
    mavlink_dnn_recv_t packet;
    packet.recovered_signal = recovered_signal;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_RECV, (const char *)&packet, MAVLINK_MSG_ID_DNN_RECV_MIN_LEN, MAVLINK_MSG_ID_DNN_RECV_LEN, MAVLINK_MSG_ID_DNN_RECV_CRC);
#endif
}

/**
 * @brief Send a dnn_recv message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_dnn_recv_send_struct(mavlink_channel_t chan, const mavlink_dnn_recv_t* dnn_recv)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_dnn_recv_send(chan, dnn_recv->recovered_signal);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_RECV, (const char *)dnn_recv, MAVLINK_MSG_ID_DNN_RECV_MIN_LEN, MAVLINK_MSG_ID_DNN_RECV_LEN, MAVLINK_MSG_ID_DNN_RECV_CRC);
#endif
}

#if MAVLINK_MSG_ID_DNN_RECV_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_dnn_recv_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  float recovered_signal)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_float(buf, 0, recovered_signal);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_RECV, buf, MAVLINK_MSG_ID_DNN_RECV_MIN_LEN, MAVLINK_MSG_ID_DNN_RECV_LEN, MAVLINK_MSG_ID_DNN_RECV_CRC);
#else
    mavlink_dnn_recv_t *packet = (mavlink_dnn_recv_t *)msgbuf;
    packet->recovered_signal = recovered_signal;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_DNN_RECV, (const char *)packet, MAVLINK_MSG_ID_DNN_RECV_MIN_LEN, MAVLINK_MSG_ID_DNN_RECV_LEN, MAVLINK_MSG_ID_DNN_RECV_CRC);
#endif
}
#endif

#endif

// MESSAGE DNN_RECV UNPACKING


/**
 * @brief Get field recovered_signal from dnn_recv message
 *
 * @return  output gyro signal
 */
static inline float mavlink_msg_dnn_recv_get_recovered_signal(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  0);
}

/**
 * @brief Decode a dnn_recv message into a struct
 *
 * @param msg The message to decode
 * @param dnn_recv C-struct to decode the message contents into
 */
static inline void mavlink_msg_dnn_recv_decode(const mavlink_message_t* msg, mavlink_dnn_recv_t* dnn_recv)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    dnn_recv->recovered_signal = mavlink_msg_dnn_recv_get_recovered_signal(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_DNN_RECV_LEN? msg->len : MAVLINK_MSG_ID_DNN_RECV_LEN;
        memset(dnn_recv, 0, MAVLINK_MSG_ID_DNN_RECV_LEN);
    memcpy(dnn_recv, _MAV_PAYLOAD(msg), len);
#endif
}
